// ChatGPT-style chat sessions, persisted as JSON in Application.persistentDataPath/megamind-sessions.json.
// A session's `id` doubles as the server `session_id`, so switching back to an older chat resumes Claude's history for
// it (as long as the server still holds that session). Messages are kept so the transcript can be re-rendered on
// launch / switch; cited memories keep their Citation (id, ts, caption, thumb_url) so thumbnails re-load.
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Megamind
{
    [Serializable]
    public class ChatMessage
    {
        public string role;      // "user" | "assistant" | "system"
        public string text;
        public double ts;        // unix seconds
        public List<Citation> cited = new List<Citation>();
    }

    [Serializable]
    public class ChatSession
    {
        public string id;
        public string title;
        public double created;
        public double updated;
        public List<ChatMessage> messages = new List<ChatMessage>();

        public bool IsEmpty => messages == null || messages.Count == 0;
    }

    [Serializable]
    public class SessionStoreFile
    {
        public int version = 1;
        public string activeId;
        public List<ChatSession> sessions = new List<ChatSession>();
    }

    public class SessionStore : MonoBehaviour
    {
        public string fileName = "megamind-sessions.json";
        [Tooltip("Oldest sessions beyond this count are dropped on save.")]
        public int maxSessions = 60;
        [Tooltip("Messages kept per session (oldest dropped).")]
        public int maxMessagesPerSession = 200;
        public const int TitleMaxChars = 36;

        /// <summary>Raised after any change (new / switch / append / rename / delete / load).</summary>
        public event Action Changed;

        public ChatSession Active { get; private set; }
        public IReadOnlyList<ChatSession> Sessions => _file.sessions;
        public string Path => System.IO.Path.Combine(Application.persistentDataPath, fileName);

        private SessionStoreFile _file = new SessionStoreFile();
        private bool _dirty;

        private void Awake()
        {
            Load();
            if (Active == null) NewSession();
        }

        private void LateUpdate() { if (_dirty) Save(); }
        private void OnApplicationPause(bool paused) { if (paused && _dirty) Save(); }
        private void OnApplicationQuit() { if (_dirty) Save(); }

        /// <summary>Starts a new chat and makes it active. An empty active chat is reused instead of stacking up blanks.</summary>
        public ChatSession NewSession()
        {
            if (Active != null && Active.IsEmpty) { Touch(); return Active; }
            var s = new ChatSession { id = Guid.NewGuid().ToString("N"), title = "", created = Now(), updated = Now() };
            _file.sessions.Insert(0, s);
            Active = s;
            _file.activeId = s.id;
            Touch();
            return s;
        }

        public ChatSession Find(string id) => _file.sessions.Find(s => s.id == id);

        public bool Switch(string id)
        {
            var s = Find(id);
            if (s == null || s == Active) return false;
            Active = s;
            _file.activeId = s.id;
            Touch();
            return true;
        }

        public void Rename(string id, string title)
        {
            var s = Find(id);
            if (s == null) return;
            s.title = Truncate(title);
            s.updated = Now();
            Touch();
        }

        public void Delete(string id)
        {
            var s = Find(id);
            if (s == null) return;
            _file.sessions.Remove(s);
            if (Active == s)
            {
                Active = _file.sessions.Count > 0 ? _file.sessions[0] : null;
                if (Active == null) NewSession(); else _file.activeId = Active.id;
            }
            Touch();
        }

        /// <summary>Appends to the active chat; the first user message becomes the title.</summary>
        public void Append(string role, string text, List<Citation> cited = null)
        {
            if (Active == null) NewSession();
            var m = new ChatMessage { role = role, text = text ?? "", ts = Now() };
            if (cited != null) m.cited.AddRange(cited);
            Active.messages.Add(m);
            while (Active.messages.Count > maxMessagesPerSession) Active.messages.RemoveAt(0);
            if (string.IsNullOrEmpty(Active.title) && role == "user" && !string.IsNullOrWhiteSpace(text)) Active.title = Truncate(text);
            Active.updated = Now();
            // most recent first
            _file.sessions.Remove(Active); _file.sessions.Insert(0, Active);
            Touch();
        }

        public static string Truncate(string text)
        {
            text = (text ?? "").Trim().Replace('\n', ' ');
            return text.Length <= TitleMaxChars ? text : text.Substring(0, TitleMaxChars - 1).TrimEnd() + "\u2026";
        }

        public static string DisplayTitle(ChatSession s) => string.IsNullOrEmpty(s.title) ? "New chat" : s.title;

        /// <summary>"Today" / "Yesterday" / "Earlier", by the session's last activity in local time.</summary>
        public static string Group(ChatSession s)
        {
            var day = DateTimeOffset.FromUnixTimeMilliseconds((long)(s.updated * 1000)).ToLocalTime().Date;
            var today = DateTime.Now.Date;
            if (day >= today) return "Today";
            if (day >= today.AddDays(-1)) return "Yesterday";
            return "Earlier";
        }

        public static double Now() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;

        private void Touch()
        {
            _dirty = true;
            Changed?.Invoke();
        }

        public void Load()
        {
            try
            {
                if (File.Exists(Path))
                {
                    var f = JsonUtility.FromJson<SessionStoreFile>(File.ReadAllText(Path));
                    if (f != null && f.sessions != null) _file = f;
                }
            }
            catch (Exception e) { Debug.LogWarning("[Megamind] sessions: could not read " + Path + ": " + e.Message); }
            _file.sessions.RemoveAll(s => s == null || string.IsNullOrEmpty(s.id));
            _file.sessions.Sort((a, b) => b.updated.CompareTo(a.updated));
            Active = Find(_file.activeId) ?? (_file.sessions.Count > 0 ? _file.sessions[0] : null);
            if (Active != null) _file.activeId = Active.id;
        }

        public void Save()
        {
            _dirty = false;
            // Drop blank chats (except the active one) and the oldest beyond maxSessions.
            _file.sessions.RemoveAll(s => s.IsEmpty && s != Active);
            while (_file.sessions.Count > maxSessions) _file.sessions.RemoveAt(_file.sessions.Count - 1);
            try
            {
                var dir = System.IO.Path.GetDirectoryName(Path);
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                string tmp = Path + ".tmp";
                File.WriteAllText(tmp, JsonUtility.ToJson(_file));
                if (File.Exists(Path)) File.Delete(Path);
                File.Move(tmp, Path);
            }
            catch (Exception e) { Debug.LogWarning("[Megamind] sessions: could not write " + Path + ": " + e.Message); }
        }
    }
}
