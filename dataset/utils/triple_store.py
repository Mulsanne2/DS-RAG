class TripleStore:
    def __init__(self):
        self.triples = []

    def add_triple(self, sub_query_number, subject, subject_id, obj, obj_id, relation, relation_id):
        self.triples.append({
            "sub_query_number": sub_query_number,
            "subject": subject,
            "subject_id": subject_id,
            "object": obj,
            "object_id": obj_id,
            "relation": relation,
            "relation_id": relation_id
        })

    def get_subjects(self):
        return [triple["subject"] for triple in self.triples]

    def get_objects(self):
        return [triple["object"] for triple in self.triples]
    
    def get_subject_ids(self):
        return [triple["subject_id"] for triple in self.triples]

    def get_object_ids(self):
        return [triple["object_id"] for triple in self.triples]

    def get_triples(self):
        return self.triples
    
    def to_dict(self):
        # change triple sotre into dictionary
        return {"triples": self.triples}
