// Requests the runtime permissions Megamind needs on Horizon OS:
//   horizonos.permission.HEADSET_CAMERA  (Passthrough Camera API, v74+)
//   android.permission.RECORD_AUDIO      (push-to-talk, stream 3)
// PassthroughCameraAccess itself waits until the camera permission is granted, so this can run any time.
// Put it on the "Megamind" GameObject (the scene builder does).
using System.Collections.Generic;
using UnityEngine;

namespace Megamind
{
    [DefaultExecutionOrder(-200)]
    public class MegamindPermissions : MonoBehaviour
    {
        public bool requestMicrophone = true;

        public static bool CameraGranted =>
            OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.PassthroughCameraAccess);

        public static bool MicrophoneGranted =>
            OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.RecordAudio);

        private void Start()
        {
            var wanted = new List<OVRPermissionsRequester.Permission>
            {
                OVRPermissionsRequester.Permission.PassthroughCameraAccess,
            };
            if (requestMicrophone) wanted.Add(OVRPermissionsRequester.Permission.RecordAudio);
            OVRPermissionsRequester.PermissionGranted += OnGranted;
            OVRPermissionsRequester.Request(wanted);
            Debug.Log($"[Megamind] permissions: camera={CameraGranted} mic={MicrophoneGranted} (requested {wanted.Count})");
        }

        private void OnDestroy() => OVRPermissionsRequester.PermissionGranted -= OnGranted;

        private static void OnGranted(string permissionId) => Debug.Log($"[Megamind] permission granted: {permissionId}");
    }
}
