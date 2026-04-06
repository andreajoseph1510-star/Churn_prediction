import pyrebase

firebaseConfig = {
    "apiKey": "AIzaSyBbs51VbRDgmxWrxdycRQTMXCx4tTRL6qo",
    "authDomain": "churn-prediction-with-ai.firebaseapp.com",
    "projectId": "churn-prediction-with-ai",
    "storageBucket": "churn-prediction-with-ai.firebasestorage.app",
    "messagingSenderId": "579432513955",
    "appId": "1:579432513955:web:c68cfd8f4cf2f0b7c05a16",
    "databaseURL":"https://console.firebase.google.com/u/0/project/churn-prediction-with-ai/database/churn-prediction-with-ai-default-rtdb/data/~2F"

}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()