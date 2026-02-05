from app import app
import json

def test_insights():
    with app.test_client() as client:
        response = client.get('/insights')
        data = json.loads(response.data)
        
        print("Status Code:", response.status_code)
        print("Metrics:", data.get('metrics'))
        print("Top 3 Features:")
        for f in data.get('top_features', [])[:3]:
            print(f"  - {f['feature']}: {f['importance']}")
            
        # Assertion to verify our changes stuck
        features = data.get('top_features', [])
        if features and features[0]['feature'] == "Education":
            print("\nSUCCESS: 'Education' is the top feature as expected.")
        else:
            print(f"\nFAILURE: Top feature is {features[0]['feature'] if features else 'None'}, expected 'Education'.")

if __name__ == "__main__":
    test_insights()
