using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OrbRotate : MonoBehaviour
{
    public float rotationSpeed = 0.1f; // Speed of the texture rotation
    private Material material;

    void Start()
    {
        // Get the material attached to the object's renderer
        material = GetComponent<Renderer>().material;
    }

    void Update()
    {
        // Calculate the new rotation angle based on the speed and time
        float offset = Time.time * rotationSpeed;

        // Apply the offset to the material's texture coordinates
        material.mainTextureOffset = new Vector2(offset, offset);
    }
}
