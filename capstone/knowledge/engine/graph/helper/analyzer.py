import asyncio
import threading
import uuid
from typing import List, Tuple, Dict, Any, Set
import spacy
import re
from core.schema.factory import (
    SkeletonStructure, 
    RelationStructure, 
    RelationEdge, 
    RhetoricalItem,
    RhetoricalRole
)
from core.schema.graph.graph import *
from core.util.normalize import wiki_resolver, nlp_normalize, SUBJECT_FILE
from knowledge.engine.graph.helper.taxonomy import *
    

def edge_type(s: EduNode, t: EduNode):
        type = None
        if isinstance(t, RhetoricalNode):
            name = f"{s.name}-{t.rrole.value.upper()}"
            type = "CONTENT"
            
        if isinstance(s, RhetoricalNode):
            name = f"{t.rrole.value.upper()}_ref_{t.name}"
            type = "REFERENCE"
        elif isinstance(s, ConceptNode):
            if isinstance(t, ConceptNode):
                type = "SEMANTIC"
            if isinstance(t, TopicNode) or isinstance(t, CommunityNode):
                type = "PART_OF"
        elif isinstance(s, TopicNode):
            if isinstance(t, TopicNode):
                type = "RELATED"
            elif isinstance(t, CommunityNode):
                type = "PART_OF"
                name = f"{s.name}_within_{t.name}"
        elif isinstance(s, CommunityNode) and isinstance(t, CommunityNode):
            type = "PREREQUISITE"

        if type is None:
            type = "RELATED"
        
        return type



nlp =spacy.load("en_core_web_sm", disable=["parser", "ner"])
from collections import Counter

def analyze(skeleton: SkeletonStructure, relations: RelationStructure) -> Tuple[List[Dict[str, Any]], RelationStructure]:
    
    cache = {} 
    parent_cand = Counter()
    cleaned_names = {}

    for c in skeleton.tree:
        doc = nlp(c.name)
        if len(doc) == 0: continue
        cache[c.name] = doc
        cleaned_names[c.name] = nlp_normalize(c.name, doc)
        
        root = next((token for token in doc if token.dep_ == "ROOT"), doc[-1])
        
        if root.i > 0:
            prefix_span = doc[:root.i]
            if len(prefix_span) > 0 and prefix_span[-1].pos_ not in ["ADJ", "ADV", "DET", "ADP"]:
                parent_cand[nlp_normalize(prefix_span.text.strip(), prefix_span)] += 1
        
        if root.pos_ in ["NOUN", "PROPN"]:
            parent_cand[nlp_normalize(root.lemma_.lower())] += 1

    cands = set(cleaned_names.values())
    valid_parent_cands = set(parent_cand.keys())
    cands.update(valid_parent_cands)

    def run_async_safely(coro_func, *args):
        result = []
        exc = []
        def runner():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                coro = coro_func(*args)
                result.append(loop.run_until_complete(coro))
            except Exception as e:
                exc.append(e)
            finally:
                loop.close()
        t = threading.Thread(target=runner)
        t.start()
        t.join()
        if exc:
            raise exc[0]
        return result[0]

    try:
        wiki_map = run_async_safely(wiki_resolver, list(cands), SUBJECT_FILE)
    except Exception:
        wiki_map = {}

    actions = {}
    
    def get_content(c):
        return getattr(c, "c", "") or (c.details[0].content if c.details else "")

    def check_parent (p_name):
        if not p_name: return False
        if p_name not in valid_parent_cands: return False
        info = wiki_map.get(p_name)
        if info and info.get("id"): return True
        return False

    for c in skeleton.tree:
        doc = cache.get(c.name)
        if not doc: continue

        root = next((token for token in doc if token.dep_ == "ROOT"), doc[-1])
        root_lemma = root.lemma_.lower()
        content = get_content(c)
        c_clean_name = cleaned_names[c.name]

        if root_lemma in SEMANTIC_MAP:
            parent_candidate = None
            if root.i > 0:
                temp_cand = nlp_normalize(doc[:root.i].text.strip(), doc[:root.i])
                if check_parent (temp_cand):
                    parent_candidate = temp_cand
            
            if parent_candidate:
                actions[c.name] = {
                    "type": "DOWNGRADE",
                    "parent": parent_candidate,
                    "role": SEMANTIC_MAP[root_lemma],
                    "content": content, "existing_details": c.details,
                    "clean_self": c_clean_name
                }
                continue

        parent = None
        
        if root.i > 0:
            prefix_span = doc[:root.i]
            if len(prefix_span) > 0 and prefix_span[-1].pos_ not in ["ADJ", "ADV", "DET", "ADP"]:
                topic_candidate = nlp_normalize(prefix_span.text.strip(), prefix_span)
                if check_parent (topic_candidate):
                    parent = topic_candidate

        if not parent and root.pos_ in ["NOUN", "PROPN"]:
            lem_clean = nlp_normalize(root_lemma)
            txt_clean = nlp_normalize(root.text.lower())
            if check_parent (lem_clean):
                parent = lem_clean
            elif check_parent (txt_clean):
                parent = txt_clean

        if parent and parent != c_clean_name:
            actions[c.name] = {
                "type": "HIERARCHY",
                "self": c.name,
                "clean_self": c_clean_name,
                "parent": parent,
                "relation_fwd": "PART_OF",
                "relation_rev": "INCLUDES",
                "content": content, "existing_details": c.details
            }
        else:
            actions[c.name] = {
                "type": "KEEP", "self": c.name, 
                "clean_self": c_clean_name,
                "content": content, "existing_details": c.details
            }

    concept_storage: Dict[str, Dict[str, Any]] = {}
    name_mapping = {}

    def get_or_create_concept(name, fb=""):
        w = wiki_map.get(name, {})
        raw_label = w.get("label", name)
        label = nlp_normalize(raw_label).title()
        
        def is_valid_concept(text):
            if not text or len(text.split()) < 2: return False
            if re.search(r'\d{2,}', text): return False
            doc = nlp(text)
            if doc[-1].pos_ not in ["NOUN", "PROPN"]: return False
            if len(doc) == 1 and doc[0].pos_ == "ADJ": return False
            return True

        too_long = len(label) > 60 or len(label.split()) > 8
        desc = w.get("desc", "").lower()

        if too_long:
            label: str = nlp_normalize(text=name).title()
            w = {}

        if label not in concept_storage:
            uid = w.get("id")
            
            if not uid:
                if not is_valid_concept(label):
                    return None
                uid = f"local"
            
            concept_storage[label] = {
                "id": uid,
                "name": label,
                "desc": w.get("desc") or fb,
                "status": "WIKI" if w.get("id") else "LOCAL",
                "details": []
            }
            
        return concept_storage[label]
    new_e = []

    for original_name, act in actions.items():
        
        if act["type"] == "DOWNGRADE":
            p_data = get_or_create_concept(act["parent"], "Extracted Parent Concept")
            if not p_data: continue
            try:
                new_rheto = RhetoricalItem(
                    role=act["role"], content=act["content"], confidence=3
                )
                p_data["details"].append(new_rheto)
            except Exception: pass 
            p_data["details"].extend(act["existing_details"])
            name_mapping[original_name] = p_data["name"]

        elif act["type"] == "HIERARCHY":
            node_data = get_or_create_concept(act["clean_self"], act["content"])
            if not node_data: continue
            node_data["details"].extend(act["existing_details"])
            name_mapping[original_name] = node_data["name"]
            
            p_data = get_or_create_concept(act["parent"], "General Concept")
            relationship1: str = "Source concept is a subtopic of the target concept"
            relationship2: str = "Source concept is estimately the foundation and root to the target concept"
            if p_data:
                new_e.append(RelationEdge(src=node_data["name"], tgt=p_data["name"], name=act["relation_fwd"], rel = relationship1))
                new_e.append(RelationEdge(src=p_data["name"], tgt=node_data["name"], name=act["relation_rev"], rel =relationship2))

        else:
            node_data: None | Dict[str, Any] = get_or_create_concept(act["clean_self"], act["content"])
            if not node_data: continue
            node_data["details"].extend(act["existing_details"])
            name_mapping[original_name] = node_data["name"]

    final_e = []
    for rel in relations.edges:
        s_new = name_mapping.get(rel.src)
        t_new = name_mapping.get(rel.tgt)
        if s_new and t_new and s_new != t_new:
            new_edge = rel.model_copy()
            new_edge.src = s_new
            new_edge.tgt = t_new
            final_e.append(new_edge)
    
    final_e.extend(new_e)
    node_list = list(concept_storage.values())

    return node_list, relations.model_copy(update={"edges": final_e})