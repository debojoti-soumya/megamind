// The drawer's chat list: sessions grouped Today / Yesterday / Earlier, active one highlighted, tap = switch
// (re-renders the transcript, reuses the session id on the server), long-press or "…" = Rename (Quest keyboard) / Delete.
// Entries are instantiated from the SessionEntry / SessionGroup prefabs made by Megamind > 1b.
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Megamind
{
    public class SessionListView : MonoBehaviour
    {
        public SessionStore store;
        public ChatPanel chatPanel;
        public PanelMenu menu;              // closed after a switch
        public Transform content;           // Drawer/Sessions/Content
        public RectTransform entryPrefab;   // SessionEntry: Title button + More button + Actions (Rename, Delete)
        public Text groupPrefab;            // SessionGroup: caption
        public GameObject emptyHint;        // "Earlier chats will show up here."
        public Color activeColor = MegamindTheme.AccentSoft;
        public float longPressSeconds = 0.6f;

        private readonly List<GameObject> _rows = new List<GameObject>();
        private string _expandedId;
        private TouchScreenKeyboard _keyboard;
        private string _renamingId;

        private void Start()
        {
            if (store != null)
            {
                store.Changed += Refresh;
                if (chatPanel != null && store.Active != null && !store.Active.IsEmpty) chatPanel.Render(store.Active);
            }
            Refresh();
        }

        private void OnDestroy() { if (store != null) store.Changed -= Refresh; }

        private void Update()
        {
            if (_keyboard == null) return;
            switch (_keyboard.status)
            {
                case TouchScreenKeyboard.Status.Done:
                    string text = _keyboard.text; _keyboard = null;
                    if (!string.IsNullOrWhiteSpace(text) && store != null) store.Rename(_renamingId, text);
                    _renamingId = null;
                    break;
                case TouchScreenKeyboard.Status.Canceled:
                case TouchScreenKeyboard.Status.LostFocus:
                    _keyboard = null; _renamingId = null;
                    break;
            }
        }

        public void Refresh()
        {
            foreach (var r in _rows) if (r) Destroy(r);
            _rows.Clear();
            if (store == null || content == null || entryPrefab == null) return;

            var visible = new List<ChatSession>();
            foreach (var s in store.Sessions) if (!s.IsEmpty || s == store.Active) visible.Add(s);
            if (emptyHint) emptyHint.SetActive(visible.Count == 0 || (visible.Count == 1 && visible[0].IsEmpty));

            string group = null;
            foreach (var s in visible)
            {
                if (s.IsEmpty) continue; // the blank active chat is "New chat", not a list row
                string g = SessionStore.Group(s);
                if (g != group && groupPrefab != null)
                {
                    group = g;
                    var h = Instantiate(groupPrefab, content);
                    h.text = g;
                    _rows.Add(h.gameObject);
                }
                _rows.Add(MakeRow(s));
            }
        }

        private GameObject MakeRow(ChatSession s)
        {
            var row = Instantiate(entryPrefab, content);
            bool active = store.Active == s;
            var title = row.Find("Main/Title")?.GetComponent<Button>();
            var more = row.Find("Main/More")?.GetComponent<Button>();
            var actions = row.Find("Actions")?.gameObject;
            var rename = row.Find("Actions/Rename")?.GetComponent<Button>();
            var delete = row.Find("Actions/Delete")?.GetComponent<Button>();
            if (title)
            {
                var label = title.GetComponentInChildren<Text>();
                if (label) label.text = SessionStore.DisplayTitle(s);
                if (active)
                {
                    var c = title.colors; c.normalColor = activeColor; c.highlightedColor = activeColor; c.selectedColor = activeColor; title.colors = c;
                }
                string id = s.id;
                title.onClick.AddListener(() => Select(id));
                var lp = title.gameObject.AddComponent<LongPress>();
                lp.seconds = longPressSeconds; lp.onLongPress = () => ToggleActions(id);
            }
            if (more) more.onClick.AddListener(() => ToggleActions(s.id));
            if (actions) actions.SetActive(_expandedId == s.id);
            if (rename) rename.onClick.AddListener(() => Rename(s.id));
            if (delete) delete.onClick.AddListener(() => Delete(s.id));
            return row.gameObject;
        }

        public void Select(string id)
        {
            if (store == null) return;
            _expandedId = null;
            if (store.Switch(id) && chatPanel != null) chatPanel.Render(store.Active);
            if (menu) menu.Close();
        }

        public void ToggleActions(string id)
        {
            _expandedId = _expandedId == id ? null : id;
            Refresh();
        }

        public void Rename(string id)
        {
            var s = store?.Find(id);
            if (s == null) return;
            _expandedId = null;
            if (!TouchScreenKeyboard.isSupported) { Refresh(); return; }
            _renamingId = id;
            _keyboard = TouchScreenKeyboard.Open(SessionStore.DisplayTitle(s), TouchScreenKeyboardType.Default, false, false, false, false, "Chat title");
            Refresh();
        }

        public void Delete(string id)
        {
            if (store == null) return;
            bool wasActive = store.Active != null && store.Active.id == id;
            _expandedId = null;
            store.Delete(id);
            if (wasActive && chatPanel != null) chatPanel.Render(store.Active);
        }

        /// <summary>Pointer held on a UI element for `seconds` (no drag-out) fires `onLongPress`; a normal click is left to the Button.</summary>
        public class LongPress : MonoBehaviour, IPointerDownHandler, IPointerUpHandler, IPointerExitHandler
        {
            public float seconds = 0.6f;
            public System.Action onLongPress;
            private float _down = -1f;
            private bool _fired;

            public void OnPointerDown(PointerEventData e) { _down = Time.unscaledTime; _fired = false; }
            public void OnPointerUp(PointerEventData e) { _down = -1f; }
            public void OnPointerExit(PointerEventData e) { _down = -1f; }

            private void Update()
            {
                if (_down < 0f || _fired || Time.unscaledTime - _down < seconds) return;
                _fired = true; _down = -1f;
                onLongPress?.Invoke();
            }
        }
    }
}
