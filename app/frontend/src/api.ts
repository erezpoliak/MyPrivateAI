const BASE = "http://localhost:5001";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Document {
  id: string;
  filename: string;
  title: string;
  page_count: number;
  chunk_count: number;
  status: "ingesting" | "ready" | "failed";
  error: string | null;
  created_at: string;
}

export interface Chat {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Source {
  id: number;
  message_id: string;
  chunk_index: number;
  doc_id: string;
  title: string;
  page_start: number | null;
  page_end: number | null;
  snippet: string;
}

export interface Trace {
  step: string;
  status: string;
  info: string;
}

export interface Message {
  id: string;
  chat_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  sources: Source[];
  traces: Trace[];
}

export interface ChatDetail extends Chat {
  messages: Message[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

export function uploadDocument(file: File): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  return req<Document>("/api/documents", { method: "POST", body: form });
}

export function listDocuments(): Promise<Document[]> {
  return req<Document[]>("/api/documents");
}

export function deleteDocument(docId: string): Promise<void> {
  return fetch(`${BASE}/api/documents/${docId}`, { method: "DELETE" }).then(
    (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    }
  );
}

// ---------------------------------------------------------------------------
// Chats
// ---------------------------------------------------------------------------

export function listChats(): Promise<Chat[]> {
  return req<Chat[]>("/api/chats");
}

export function createChat(): Promise<Chat> {
  return req<Chat>("/api/chats", { method: "POST" });
}

export function getChat(chatId: string): Promise<ChatDetail> {
  return req<ChatDetail>(`/api/chats/${chatId}`);
}

export function deleteChat(chatId: string): Promise<void> {
  return fetch(`${BASE}/api/chats/${chatId}`, { method: "DELETE" }).then(
    (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    }
  );
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

export function postMessage(
  chatId: string,
  content: string
): Promise<Response> {
  return fetch(`${BASE}/api/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
}
