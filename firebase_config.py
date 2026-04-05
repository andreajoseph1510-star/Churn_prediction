import pyrebase

firebaseConfig = {
    "apiKey": "AIzaSyBbs51VbRDgmxWrxdycRQTMXCx4tTRL6qo",
    "authDomain": "churn-prediction-with-ai.firebaseapp.com",
    "projectId": "churn-prediction-with-ai",
    "storageBucket": "churn-prediction-with-ai.firebasestorage.app",
    "messagingSenderId": "579432513955",
    "appId": "1:579432513955:web:c68cfd8f4cf2f0b7c05a16"

}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()