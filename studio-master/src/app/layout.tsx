// /full-stack/studio-master/src/app/layout.tsx
import type {Metadata} from 'next';
import './globals.css';
import { Toaster } from "@/components/ui/toaster";
import AnimatedLayout from '@/components/layout/AnimatedLayout'; // Import the new layout

export const metadata: Metadata = {
  title: 'VigilanceAI',
  description: 'Advanced Security Automation',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet" />
      </head>
      <body className="font-code antialiased">
        <AnimatedLayout>
          {children}
        </AnimatedLayout>
        <Toaster />
      </body>
    </html>
  );
}