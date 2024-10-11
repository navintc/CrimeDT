using UnityEngine;

public class MarkerClickHandler : MonoBehaviour
{
    private CrimeData crimeData;
    private MarkerManager markerManager;

    // Initialize with crime data and a reference to the MarkerManager
    public void Initialize(CrimeData crime, MarkerManager manager)
    {
        this.crimeData = crime;
        this.markerManager = manager;
    }

    private void OnMouseDown()
    {
        // When the marker is clicked, call DisplayCrimeInstructions on the MarkerManager
        if (markerManager != null && crimeData != null)
        {
            markerManager.DisplayCrimeInstructions(crimeData);
        }
    }
}
