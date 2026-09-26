from locust import HttpUser, task


class Predict(HttpUser):
    @task
    def predict(self):
        self.client.put(
            "/predict", json={"prompt": "The quick brown fox jumps over the lazy dog"}
        )
