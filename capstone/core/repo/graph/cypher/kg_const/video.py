CYPHER_write_vid = """
MERGE (v:Video {id: $vid.id})
SET v.title=$vid.title, v.uri=$vid.uri, v.duration=$vid.duration
"""

CYPHER_write_segs = """
UNWIND $segments AS s
MERGE (n:Segment {id: s.id})
SET n.t_lo=s.p_num[0], n.t_hi=s.p_num[1], n.text=s.text, n.emb=s.emb,
    n.uri=$uri, n.order=s.order, n.best_score=s.best_score,
    n.candidate_ids=s.candidate_ids, n.candidate_names=s.candidate_names,
    n.candidate_scores=s.candidate_scores, n.anchor_model=s.anchor_model,
    n.anchor_version=s.anchor_version, n.novel_candidate=s.novel_candidate
REMOVE n.anchor_scores
WITH n
MATCH (v:Video {id: $vid_id})
MERGE (v)-[:HAS_SEGMENT]->(n)
"""

CYPHER_get_segment_context = """
MATCH (v:Video)-[:HAS_SEGMENT]->(target:Segment {id: $sid})
MATCH (v)-[:HAS_SEGMENT]->(s:Segment)
WITH s, target ORDER BY s.t_lo
RETURN s.id AS id, s.t_lo AS t_lo, s.t_hi AS t_hi, s.text AS text,
       (s.id = target.id) AS is_target
"""
