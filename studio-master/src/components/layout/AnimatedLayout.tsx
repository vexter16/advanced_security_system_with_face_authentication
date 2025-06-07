// /full-stack/studio-master/src/components/layout/AnimatedLayout.tsx
'use client';

import { useRef, useEffect, useCallback } from 'react';

export default function AnimatedLayout({ children }: { children: React.ReactNode }) {
  const starfieldCanvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameId = useRef<number>();

  const runAnimation = useCallback(() => {
    const starfieldCanvas = starfieldCanvasRef.current;
    if (!starfieldCanvas) return;
    const s_ctx = starfieldCanvas.getContext('2d');
    if (!s_ctx) return;

    starfieldCanvas.width = window.innerWidth;
    starfieldCanvas.height = window.innerHeight;

    const stars = Array.from({ length: 1500 }, () => ({
      x: Math.random() * starfieldCanvas.width,
      y: Math.random() * starfieldCanvas.height,
      z: Math.random() * starfieldCanvas.width,
    }));

    const animate = () => {
      s_ctx.fillStyle = "hsl(var(--background))";
      s_ctx.fillRect(0, 0, starfieldCanvas.width, starfieldCanvas.height);
      
      stars.forEach(star => {
        star.z -= 1.5;
        if (star.z <= 0) {
          star.z = starfieldCanvas.width;
        }
        const k = 128 / star.z;
        const px = star.x * k + starfieldCanvas.width / 2;
        const py = star.y * k + starfieldCanvas.height / 2;
        if (px > 0 && px < starfieldCanvas.width && py > 0 && py < starfieldCanvas.height) {
          const size = ((1 - star.z / starfieldCanvas.width) * 2.5);
          s_ctx.beginPath();
          s_ctx.fillStyle = Math.random() > 0.95 ? 'hsl(var(--primary))' : 'hsl(var(--secondary))';
          s_ctx.fillRect(px, py, size, size); // Use squares for a more digital look
        }
      });
      animationFrameId.current = requestAnimationFrame(animate);
    };
    animate();
  }, []);
  
  useEffect(() => {
    runAnimation();
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [runAnimation]);

  return (
    <>
      <canvas ref={starfieldCanvasRef} id="persistent-background-canvas" />
      <div className="main-content-wrapper">
        {children}
      </div>
      <style jsx global>{`
        #persistent-background-canvas {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: -1; /* Place it behind everything */
        }
        .main-content-wrapper {
          position: relative;
          z-index: 1;
          width: 100%;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
        }
      `}</style>
    </>
  );
}