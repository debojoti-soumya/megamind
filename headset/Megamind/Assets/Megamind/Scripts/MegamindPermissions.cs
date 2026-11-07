// Single runtime-permission flow for Megamind (mirrors Meta's PassthroughCameraApiSamples RequestPermissionsOnce):
// one OVRPermissionsRequester.Request(...) call, once per process, for
//   horizonos.permission.HEADSET_CAMERA  (Passthrough Camera API, v74+; OVRPermissionsRequester.Permission.PassthroughCameraAccess)
//   android.permission.RECORD_AUDIO      (push-to-talk, stream 3)
// Nothing else may request camera permission: keep OVRManager "Permission Requests On Startup" empty (the scene builder
// clears it) and do not add the sample's PassthroughCameraPermissions/RequestPermissionsOnce to the scene.
// PassthroughCameraAccess itself waits until the permission is granted, so this can run any time before/after it.
using System.Collections.Generic;
using UnityEngine;

namespace Megamind
{
    [DefaultExecutionOrder(-200)]
    public class MegamindPermissions : MonoBehaviour
    {
        public bool requestMicrophone = true;

        private static bool s_requested;

        public static bool CameraGranted =>
            OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.PassthroughCameraAccess);

        public static bool MicrophoneGranted =>
            OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.RecordAudio);

        private void Start()
        {
            OVRPermissionsRequester.PermissionGranted += OnGranted;
            if (s_requested)
            {
                Debug.Log("[Megamind] permissions: already requested this session; skipping");
                return;
            }
            s_requested = true;
            var wanted = new List<OVRPermissionsRequester.Permission>();
            if (!CameraGranted) wanted.Add(OVRPermissionsRequester.Permission.PassthroughCameraAccess);
            if (requestMicrophone && !MicrophoneGranted) wanted.Add(OVRPermissionsRequester.Permission.RecordAudio);
            if (wanted.Count > 0) OVRPermissionsRequester.Request(wanted);
            Debug.Log($"[Megamind] permissions: camera={CameraGranted} mic={MicrophoneGranted} (requested {wanted.Count})");
        }

        private void OnDestroy() => OVRPermissionsRequester.PermissionGranted -= OnGranted;

        private static void OnGranted(string permissionId) => Debug.Log($"[Megamind] permission granted: {permissionId}");
    }
}
