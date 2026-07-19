CYPHER_write_video = """
MERGE (v:Video {id: $video.id})
SET v.title=$video.title, v.uri=$video.uri, v.duration=$video.duration
"""

CYPHER_write_segments = """
UNWIND $segments AS s
MERGE (n:Segment {id: s.id})
SET n.t_lo=s.p_num[0], n.t_hi=s.p_num[1], n.text=s.text, n.emb=s.emb,
    n.uri=$uri, n.order=s.order, n.best_score=s.best_score,
    n.anchor_scores=s.anchor_scores, n.novel_candidate=s.novel_candidate
WITH n
MATCH (v:Video {id: $video_id})
MERGE (v)-[:HAS_SEGMENT]->(n)
"""

CYPHER_get_segment_context = """
MATCH (v:Video)-[:HAS_SEGMENT]->(target:Segment {id: $sid})
MATCH (v)-[:HAS_SEGMENT]->(s:Segment)
WITH s, target ORDER BY s.t_lo
RETURN s.id AS id, s.t_lo AS t_lo, s.t_hi AS t_hi, s.text AS text,
       (s.id = target.id) AS is_target
"""
