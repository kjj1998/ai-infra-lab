from locust import FastHttpUser, task


class Predict(FastHttpUser):
    @task
    def predict(self):
        self.client.post(
            "/predict", json={"prompt": "The quick brown fox jumps over the lazy dog"}
        )
