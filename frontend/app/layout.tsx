import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Autonomous Oncology Board | AMD MI300X",
  description:
    "3-agent AI tumour board running GigaPath + Llama 3.3 70B on AMD MI300X 192GB HBM3. Multi-round agent debate, NCCN 2024 guidelines, attention heatmaps.",
  keywords: ["oncology", "AI", "AMD MI300X", "GigaPath", "medical AI", "tumour board"],
  openGraph: {
    title: "Autonomous Oncology Board",
    description: "3-agent AI tumour board on AMD MI300X — GigaPath + Llama 3.3 70B",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ background: "var(--bg-base)", color: "var(--text-primary)" }}>
        {children}
      </body>
    </html>
  );
}
