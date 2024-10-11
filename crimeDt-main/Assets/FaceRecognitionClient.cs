using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

public class FaceRecognition : MonoBehaviour
{
    public string apiUrl = "http://127.0.0.1:5001/face/recognize";  // URL to your Flask API
    public RawImage faceDisplay;  // UI element to display the face image

    void Start()
    {
        StartCoroutine(GetFaceData());
    }

    IEnumerator GetFaceData()
{
    int attempts = 0;
    while (attempts < 300) // Try 3 times
    {
        UnityWebRequest request = UnityWebRequest.Get(apiUrl);
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.LogError(request.error);
            attempts++;
            yield return new WaitForSeconds(2); // Wait before retrying
        }
        else
        {
            // Log the raw response for debugging
            string jsonResponse = request.downloadHandler.text;
            Debug.Log("Raw JSON Response: " + jsonResponse);

            // Trim whitespace and check for valid JSON
            jsonResponse = jsonResponse.Trim();

            // Make sure the response is valid JSON
            if (jsonResponse.StartsWith("{") && jsonResponse.EndsWith("}"))
            {
                // Parse the JSON response
                FaceRecognitionResponse response = JsonUtility.FromJson<FaceRecognitionResponse>(jsonResponse);

                // Check if any faces are found
                if (response.faces != null && response.faces.Length > 0 && response.images != null && response.images.Length > 0)
                {
                    // Load the image from the path
                    StartCoroutine(LoadFaceImage(response.images[0]));  // Load the first matching image
                    yield break; // Exit after successful retrieval
                }
                else
                {
                    Debug.Log("No matching image found");
                }
            }
            else
            {
                Debug.LogError("Invalid JSON format: " + jsonResponse);
            }
        }
    }
    Debug.LogError("Failed to connect after multiple attempts.");
}

    IEnumerator LoadFaceImage(string imagePath)
    {
        // Convert file path to a URL (file://)
        string filePath = "file://" + imagePath;
        UnityWebRequest imageRequest = UnityWebRequestTexture.GetTexture(filePath);

        yield return imageRequest.SendWebRequest();

        if (imageRequest.result == UnityWebRequest.Result.ConnectionError || imageRequest.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.LogError(imageRequest.error);
        }
        else
        {
            // Get the texture from the request
            Texture2D texture = DownloadHandlerTexture.GetContent(imageRequest);

            // Assign the texture to the RawImage to display it
            faceDisplay.texture = texture;
        }
    }
}

[System.Serializable]
public class FaceRecognitionResponse
{
    public string[] faces;  // Array to store face names
    public string[] images; // Array to store image paths
}
