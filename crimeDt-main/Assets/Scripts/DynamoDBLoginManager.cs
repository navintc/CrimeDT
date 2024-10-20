using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using Newtonsoft.Json;

public class DynamoDBLoginManager : MonoBehaviour
{
    // URL of your Flask server running the login handler
    private string flaskServerUrl = "http://127.0.0.1:5001";

    [SerializeField]
    public InputField emailInputField;
    [SerializeField]
    public InputField passwordInputField;
    [SerializeField]
    public Text resultText;

    // Method to call Flask server for login
    public void Login()
    {
        string email = emailInputField.text;
        string password = passwordInputField.text;

        if (string.IsNullOrEmpty(email) || string.IsNullOrEmpty(password))
        {
            resultText.text = "Email and Password fields cannot be empty.";
            return;
        }

        StartCoroutine(LoginCoroutine(email, password));
    }

    private IEnumerator LoginCoroutine(string email, string password)
    {
        // Create a JSON object to send as part of the request
        var loginData = new { email = email, password = password };
        string jsonData = JsonConvert.SerializeObject(loginData);

        // Create the web request to the Flask login endpoint
        UnityWebRequest request = new UnityWebRequest(flaskServerUrl + "/internallogin", "POST")
        {
            uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(jsonData)),
            downloadHandler = new DownloadHandlerBuffer()
        };

        request.SetRequestHeader("Content-Type", "application/json");

        // Wait for the request to complete
        yield return request.SendWebRequest();

        // Handle errors
        if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.LogError(request.error);
            resultText.text = "Login failed: " + request.error;
        }
        else
        {
            // Parse the response from the Flask server
            string responseJson = request.downloadHandler.text;
            var loginResponse = JsonConvert.DeserializeObject<LoginResponse>(responseJson);

            if (loginResponse.message == "Login successful")
            {
                resultText.text = "Login successful! Welcome, " + loginResponse.name;
                // Handle successful login logic here
            }
            else
            {
                resultText.text = "Login failed: " + loginResponse.message;
            }
        }
    }

    [System.Serializable]
    public class LoginResponse
    {
        public bool success;
        public string name;
        public string message;
    }
}
