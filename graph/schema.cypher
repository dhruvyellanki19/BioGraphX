// Neo4j Graph Schema for BioGraphX MedQuAD Graph
// --------------------------------------------------

// Node Constraints
CREATE CONSTRAINT question_id IF NOT EXISTS
FOR (q:Question) REQUIRE q.qid IS UNIQUE;

CREATE CONSTRAINT disease_name IF NOT EXISTS
FOR (d:Disease) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT chemical_name IF NOT EXISTS
FOR (c:Chemical) REQUIRE c.name IS UNIQUE;


// Relationship Types (documentation only)
// ----------------------------------------

// (Question)-[:MENTIONS_DISEASE]->(Disease)
// Properties: none

// (Question)-[:MENTIONS_CHEMICAL]->(Chemical)
// Properties: none


// Sample Cypher for Data Import (using neo4j-admin import)
// ---------------------------------------------------------

// LOAD CSV WITH HEADERS FROM 'file:///medquad_nodes_questions.csv' AS row
// CREATE (:Question {
//     qid: row.qid,
//     question: row.question,
//     answer: row.answer,
//     focus_area: row.focus_area,
//     source: row.source
// });

// LOAD CSV WITH HEADERS FROM 'file:///medquad_nodes_diseases.csv' AS row
// CREATE (:Disease {name: row.name, source: row.source});

// LOAD CSV WITH HEADERS FROM 'file:///medquad_nodes_chemicals.csv' AS row
// CREATE (:Chemical {name: row.name, source: row.source});

// LOAD CSV WITH HEADERS FROM 'file:///medquad_edges_mentions_disease.csv' AS row
// MATCH (q:Question {qid: row.qid})
// MATCH (d:Disease {name: row.disease_name})
// MERGE (q)-[:MENTIONS_DISEASE]->(d);

// LOAD CSV WITH HEADERS FROM 'file:///medquad_edges_mentions_chemical.csv' AS row
// MATCH (q:Question {qid: row.qid})
// MATCH (c:Chemical {name: row.chemical_name})
// MERGE (q)-[:MENTIONS_CHEMICAL]->(c);
