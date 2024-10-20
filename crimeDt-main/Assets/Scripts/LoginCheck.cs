using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;
using UnityEngine.UI;
using UnityEngine;


[System.Serializable]
public class UserInfo
{
    public string aud;
    public string email;
    public bool email_verified;
    public long exp;
    public long iat;
    public string iss;
    public string name;
    public string nickname;
    public string nonce;
    public string picture;
    public string sid;
    public string sub;
    public string updated_at;

    public override string ToString()
    {
        return $"Name: {name}, Email: {email}";
    }
}

[System.Serializable]
public class UserData
{
    public string access_token;
    public long expires_at;
    public long expires_in;
    public string id_token;
    public string scope;
    public string token_type;
    public UserInfo userinfo;

    public override string ToString()
    {
        return $"Access Token: {access_token}, User Info: [{userinfo}]";
    }
}

[System.Serializable]
public class TokenResponse
{
    public string message;
    public UserData user_data;

    public override string ToString()
    {
        return $"Message: {message}, User Data: [{user_data}]";
    }
}

public class LoginCheck : MonoBehaviour
{
    
    public string nickname;
    public string userEmail;

    [SerializeField]
    public InputField tokenInputField;

    [SerializeField]
    public Button checkTokenButton;

    private string flaskServerUrl = "http://127.0.0.1:5001/check_token";

    void Start()
    {
        // Assign the listener to the button
        checkTokenButton.onClick.AddListener(OnCheckTokenButtonClick);
    }

    // Called when the "Check Token" button is clicked
    public void OnCheckTokenButtonClick()
    {
        string token = tokenInputField.text;

        if (string.IsNullOrEmpty(token))
        {
            Debug.LogError("Token is required");
            return;
        }

        StartCoroutine(CheckTokenRequest(token));
    }

    // Coroutine to make the POST request to Flask server to check the token
    private IEnumerator CheckTokenRequest(string token)
    {
        // Create a JSON object with the token
        string json = "{\"token\":\"" + token + "\"}";
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        // Create the UnityWebRequest for the POST request
        UnityWebRequest request = new UnityWebRequest(flaskServerUrl, "POST");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        // Send the request
        yield return request.SendWebRequest();

        // Handle the response
        if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.LogError("Error: " + request.error);
            Debug.LogError("Response Code: " + request.responseCode);
        }
        else
        {
            // Successful response, process the JSON response
            string jsonResponse = request.downloadHandler.text;
            Debug.Log("Response: " + jsonResponse);

            // Parse the JSON to get the email and name
            TokenResponse response = JsonUtility.FromJson<TokenResponse>(jsonResponse);

            if (response != null && response.user_data != null)
            {
                // Log the parsed data
                Debug.Log("Parsed Response: " + response.ToString());
                Debug.Log("Parsed UserData: " + response.user_data.ToString());
                Debug.Log("Parsed UserInfo: " + response.user_data.userinfo.ToString());

                // Assign to public variables for future use
                nickname = response.user_data.userinfo.nickname;
                userEmail = response.user_data.userinfo.email;

                // Log individual properties
                Debug.Log("User Name: " + nickname);
                Debug.Log("User Email: " + userEmail);
            }
            else
            {
                Debug.LogError("Unable to parse user information from the response.");
            }
        }
    }
}
