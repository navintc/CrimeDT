using UnityEngine;
using CesiumForUnity;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine.Networking;  // For making HTTP requests
using System.Text;
using Newtonsoft.Json;  // Use Newtonsoft.Json for JSON handling
using UnityEngine.UI;  // Required for UI elements
using TMPro;
using System.Linq;

public class MarkerManager : MonoBehaviour
{
    public GameObject markerPrefab; // Reference to the marker prefab
    public CesiumGeoreference cesiumGeoreference; // Reference to the CesiumGeoreference component
    public string apiUrl = "http://localhost:5001/lstm/predict"; // URL of your Flask API endpoint
    private string openAiApiKey = ""; // OpenAI API Key

    public TMP_Text instructionsText; // Reference to a UI Text component to display instructions
    private Dictionary<string, string> crimeInstructionsCache = new Dictionary<string, string>(); // Cache for storing instructions per location (keyed by a Lat-Lon string)

    private void Start()
    {
        StartCoroutine(FetchCoordinatesFromAPI());
    }

    private IEnumerator FetchCoordinatesFromAPI()
    {
        using (UnityWebRequest request = UnityWebRequest.Get(apiUrl))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Error fetching data from API: " + request.error);
                yield break;
            }

            // Parse the JSON response
            string jsonResponse = request.downloadHandler.text;

            // Log the raw JSON response for debugging
            Debug.Log("Raw JSON Response: " + jsonResponse);

            // Verify if JSON is valid
            if (string.IsNullOrEmpty(jsonResponse) || jsonResponse[0] != '[')
            {
                Debug.LogError("Invalid JSON response: " + jsonResponse);
                yield break;
            }

            // Parse the JSON response into a list of CrimeData objects
            List<CrimeData> crimeDataList = JsonHelper.FromJson<CrimeData>(jsonResponse);

            // Log the number of crimes fetched
            Debug.Log($"Fetched {crimeDataList.Count} crime data entries.");

            // Log each crime entry details
            foreach (CrimeData crime in crimeDataList)
            {
                Debug.Log($"Crime Type: {crime.PredictedCrimeType}, " +
                          $"Location: Latitude {crime.Latitude}, Longitude {crime.Longitude}, " +
                          $"Date: {crime.Day}/{crime.Month}/{crime.Year}, " +
                          $"Time Range: {crime.PredictedTime}, " +
                          $"Probability: {crime.Probability}");
            }

            // Place markers at the coordinates
            foreach (CrimeData crime in crimeDataList)
            {
                double longitude = crime.Longitude;
                double latitude = crime.Latitude;
                double altitude = 350;  // Use 0 or fetch altitude if available

                // Instantiate the marker prefab
                GameObject marker = Instantiate(markerPrefab);

                // Ensure marker is a child of the CesiumGeoreference
                marker.transform.parent = cesiumGeoreference.transform;

                // Add CesiumGlobeAnchor component to manage geospatial positioning
                CesiumGlobeAnchor anchor = marker.AddComponent<CesiumGlobeAnchor>();

                // Set the geodetic position
                anchor.longitudeLatitudeHeight = new double3(longitude, latitude, altitude);

                // Log the position for debugging
                Debug.Log($"Placing marker at Longitude: {longitude}, Latitude: {latitude}, Altitude: {altitude}");

                // Add a collider to the marker to detect clicks
                marker.AddComponent<SphereCollider>().radius = 0.5f; // Adjust the radius as needed
                marker.AddComponent<MarkerClickHandler>().Initialize(crime, this); // Attach click handler script and pass crime data
            }

            // After placing all markers, generate instructions for all crimes in a single request
            yield return StartCoroutine(GenerateInstructionsForAllCrimes(crimeDataList));
        }
    }

    // Function to generate instructions using OpenAI API (ChatGPT) for all predicted crimes
    private IEnumerator GenerateInstructionsForAllCrimes(List<CrimeData> crimes)
    {
        const int batchSize = 3; // Number of crimes per request
        for (int i = 0; i < crimes.Count; i += batchSize)
        {
            var batch = crimes.Skip(i).Take(batchSize).ToList(); // Get the current batch of crimes

            // Create a single prompt with details of the current batch of predicted crimes
            StringBuilder promptBuilder = new StringBuilder();
            promptBuilder.AppendLine("Provide detailed police instructions for handling the following predicted crimes:");

            foreach (CrimeData crime in batch)
            {
                promptBuilder.AppendLine($"- Crime at coordinates Latitude: {crime.Latitude}, Longitude: {crime.Longitude}, " +
                                         $"predicted for {crime.PredictedTime} on {crime.Day}/{crime.Month}/{crime.Year}. Crime Type: {crime.PredictedCrimeType}.");
            }

            // Build the final prompt
            string prompt = promptBuilder.ToString();

            var requestBody = new
            {
                model = "gpt-4o",
                messages = new List<object>
                {
                    // new { role = "system", content = "You are an assistant that generates police instructions for predicted crimes. Make sure to clearly label the instructions with the coordinates for each crime according to the given hours of the day." },
                    new { role = "system", content = "generate police instructions for predicted crimes, labeling them with coordinates based on the time of day. Provide specific advice considering a radius of 500m geography (incl. rivers, lakes) around the given longitude and latitude."},
                    new { role = "user", content = prompt }
                }
            };

            string jsonRequestBody = JsonConvert.SerializeObject(requestBody);
            Debug.Log("This is a log message."+ jsonRequestBody);

            using (UnityWebRequest request = new UnityWebRequest("https://api.openai.com/v1/chat/completions", "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonRequestBody);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                request.SetRequestHeader("Authorization", $"Bearer {openAiApiKey}");

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string responseText = request.downloadHandler.text;
                    OpenAIResponse response = JsonConvert.DeserializeObject<OpenAIResponse>(responseText);

                    // Assuming the response contains instructions for all crimes in the batch
                    string instructions = response.choices[0].message.content;
                    Debug.Log("Generated instructions for batch: " + instructions);

                    // Cache the generated instructions for each crime in the batch
                    CacheInstructions(batch, instructions);
                }
                else
                {
                    Debug.LogError("Error calling OpenAI API: " + request.error);
                    Debug.LogError("Response Text: " + request.downloadHandler.text);
                }
            }
        }
    }

    // Cache the instructions for each crime based on its latitude and longitude
    private void CacheInstructions(List<CrimeData> crimes, string fullInstructions)
    {
        // Split the full instructions into individual crime instructions based on crime location
        foreach (CrimeData crime in crimes)
        {
            string key = $"{crime.Latitude},{crime.Longitude}";

            // Find the instructions that contain the corresponding latitude and longitude
            string crimePattern = $"Latitude: {crime.Latitude}, Longitude: {crime.Longitude}";
            int startIndex = fullInstructions.IndexOf(crimePattern);
            if (startIndex != -1)
            {
                // Find the next crime or end of the string to get this crime's specific instructions
                int endIndex = fullInstructions.IndexOf("Latitude:", startIndex + crimePattern.Length);
                if (endIndex == -1) endIndex = fullInstructions.Length; // If it's the last crime in the list

                string crimeInstructions = fullInstructions.Substring(startIndex, endIndex - startIndex).Trim();

                // Cache the instructions for this specific crime
                if (!crimeInstructionsCache.ContainsKey(key))
                {
                    // Remove redundant details from the instructions if needed
                    crimeInstructionsCache.Add(key, crimeInstructions);
                }
            }
        }
    }

    // Method to display crime instructions on the UI when a marker is clicked
    public void DisplayCrimeInstructions(CrimeData crime)
    {
        if (instructionsText != null)
        {
            string key = $"{crime.Latitude},{crime.Longitude}";

            if (crimeInstructionsCache.TryGetValue(key, out string instructions))
            {
                // Set the text in the UI
                instructionsText.text = $"Crime: {crime.PredictedCrimeType}\n" +
                                        $"Instructions:\n{instructions}\n";
            }
            else
            {
                instructionsText.text = "No instructions available for this crime.";
            }
        }
        else
        {
            Debug.LogError("Instructions Text is not assigned in the Inspector.");
        }
    }
}

// Helper class for OpenAI response
[System.Serializable]
public class OpenAIResponse
{
    public Choice[] choices;

    [System.Serializable]
    public class Choice
    {
        public Message message;
    }

    [System.Serializable]
    public class Message
    {
        public string content;
    }
}

// Helper class to represent the data from the API
[System.Serializable]
public class CrimeData
{
    public double Longitude;
    public double Latitude;

    [JsonProperty("Predicted CrimeType")] // Ensure this matches your JSON output
    public string PredictedCrimeType;

    public float Probability;
    public int Year;
    public int Month;
    public int Day;

    // New field for predicted time range
    [JsonProperty("Predicted Time")] // Ensure this matches your JSON output
    public string PredictedTime; // Example: "14:00 - 15:00" or "Between 2 PM and 3 PM"
}

// Helper class for JSON parsing
public static class JsonHelper
{
    public static List<T> FromJson<T>(string json)
    {
        // Remove whitespace characters
        json = json.Trim();
        if (json.StartsWith("[") && json.EndsWith("]"))
        {
            json = "{\"array\":" + json + "}";
        }

        Wrapper<T> wrapper = JsonConvert.DeserializeObject<Wrapper<T>>(json);
        return new List<T>(wrapper.array);
    }

    [System.Serializable]
    private class Wrapper<T>
    {
        public T[] array;
    }
}
