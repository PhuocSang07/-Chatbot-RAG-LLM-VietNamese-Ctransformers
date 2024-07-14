import numpy as np

class Router:
    def __init__(self, embedding, routes):
        self.embedding = embedding
        self.routes = routes
        self.routes_embedding = {}

        for route in self.routes:
            self.routes_embedding[route.name] = self.embedding.encode(route.sample)

    def get_routes(self):
        return self.routes

    def route(self, query):
        query_embedding = self.embedding.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        similarity_score = []

        for route in self.routes:
            routers_embedding = self.routes_embedding[route.name] / np.linalg.norm(self.routes_embedding[route.name])
            similarity = np.mean(np.dot(query_embedding, routers_embedding.T).flatten())
            similarity_score.append(similarity, route.name)

        similarity_score.sort(reverse=True) 

        return similarity_score[0]