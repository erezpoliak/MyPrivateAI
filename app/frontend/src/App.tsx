import { useState } from "react";
import { Toaster } from "sonner";
import Documents from "./pages/Documents";
import Chat from "./pages/Chat";

type Tab = "documents" | "chat";

const TABS: { id: Tab; label: string }[] = [
  { id: "documents", label: "Documents" },
  { id: "chat", label: "Chat" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("documents");

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <header className="border-b border-gray-200">
        <nav className="flex gap-1 px-4">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={[
                "px-4 py-3 text-sm font-medium border-b-2 -mb-px transition-colors",
                activeTab === tab.id
                  ? "border-indigo-600 text-indigo-600"
                  : "border-transparent text-gray-500 hover:text-gray-700",
              ].join(" ")}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1">
        {activeTab === "documents" ? <Documents /> : <Chat />}
      </main>

      <Toaster position="bottom-right" richColors />
    </div>
  );
}
