using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;
using UnityEngine;

public class LoginManager : MonoBehaviour
{
     private string flaskServerUrl = "http://127.0.0.1:5001";

    public void Login()
    {
        Application.OpenURL(flaskServerUrl + "/login");
    }

    public void OnLoginCompletedButtonClick()
    {
        StartCoroutine(GetUserData());
    }
    

    public IEnumerator GetUserData()
    {
        UnityWebRequest request = UnityWebRequest.Get(flaskServerUrl + "/dashboard");
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.LogError(request.error);
            Debug.LogError("Response Code: " + request.responseCode);  // Logs the HTTP response code
            Debug.LogError("Error Response: " + request.downloadHandler.text);  // Logs the content of the response (if any)
        }
        else
        {
            Debug.Log("User Data: " + request.downloadHandler.text);
        }
    }

    public void Logout()
    {
        Application.OpenURL(flaskServerUrl + "/logout");
    }
}
