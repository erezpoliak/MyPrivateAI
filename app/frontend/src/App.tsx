import { useState } from "react";
import { Toaster } from "sonner";
import Documents from "./pages/Documents";
import Chat from "./pages/Chat";
import styles from "./App.module.css";

type Tab = "library" | "workspace";

export default function App() {
  const [tab, setTab] = useState<Tab>("library");

  return (
    <div style={{ height: "100svh", display: "flex", flexDirection: "column", overflow: "hidden", background: "var(--bg)" }}>
      <header className={styles.header}>
        <span className={styles.logo}>MyPrivateAI</span>

        <div className={styles.spacer} />

        <div className={styles.tabGroup}>
          {([ ["library", "Library"], ["workspace", "Workspace"] ] as const).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={styles.tab}
              data-active={tab === id}
            >
              {label}
            </button>
          ))}
        </div>

        <div className={styles.avatar}>USR</div>
      </header>

      <main className={styles.main}>
        {tab === "library" ? <Documents /> : <Chat />}
      </main>

      <Toaster position="bottom-right" richColors />
    </div>
  );
}
