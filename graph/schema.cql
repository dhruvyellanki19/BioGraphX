CREATE CONSTRAINT question_id_unique IF NOT EXISTS
FOR (q:Question)
REQUIRE q.id IS UNIQUE;

CREATE CONSTRAINT disease_name_unique IF NOT EXISTS
FOR (d:Disease)
REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
FOR (d:Drug)
REQUIRE d.name IS UNIQUE;
