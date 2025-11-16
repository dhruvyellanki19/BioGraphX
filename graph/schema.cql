// --- Question ---
CREATE CONSTRAINT question_id_unique IF NOT EXISTS
FOR (q:Question)
REQUIRE q.id IS UNIQUE;

// --- Disease ---
CREATE CONSTRAINT disease_name_unique IF NOT EXISTS
FOR (d:Disease)
REQUIRE d.name IS UNIQUE;

// --- Drug / Chemical ---
CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
FOR (d:Drug)
REQUIRE d.name IS UNIQUE;

// Relationship pattern:
// (:Question)-[:ABOUT]->(:Disease | :Drug)
