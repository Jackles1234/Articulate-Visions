import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css"; // Import global styles here
import { Analytics } from "@vercel/analytics/react" // Vercel Analytics

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Articulate Visions",
  description: "Visualizing diffusion text to image models",
};

export default function RootLayout({
                                     children,
                                   }: Readonly<{
  children: React.ReactNode;
}>) {
  return (
      <html lang="en">
      <body className={inter.className}>{children}
      <Analytics />
      </body>

      </html>
  );
}