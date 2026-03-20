import type { Metadata } from "next";
import { Merriweather, Inter } from "next/font/google";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Analytics } from "@vercel/analytics/react";
import "./globals.css";
import "highlight.js/styles/github.min.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const merriweather = Merriweather({
  subsets: ["latin"],
  weight: ["300", "400", "700", "900"],
  variable: "--font-serif", 
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans", 
});

export const metadata: Metadata = {
  title: "HumanStudy-Bench: Towards AI Agent Design for Participant Simulation",
  description: "A reusable platform for replaying human-subject experiments end-to-end, evaluating agent design alignment with human ground-truth outcomes at the level of scientific inference.",
  icons: {
    icon: [{ url: "/favicon.svg?v=2", type: "image/svg+xml" }],
    shortcut: "/favicon.svg?v=2",
    apple: "/favicon.svg?v=2",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${merriweather.variable} ${inter.variable} font-sans antialiased min-h-screen flex flex-col bg-white text-black`}
      >
        <Navbar />
        <main className="flex-grow">{children}</main>
        <Footer />
        <SpeedInsights />
        <Analytics />
      </body>
    </html>
  );
}
