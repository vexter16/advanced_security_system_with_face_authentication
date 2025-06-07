// /full-stack/studio-master/src/app/page.tsx
'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import LockScreenForm from '@/components/auth/LockScreenForm';

// A simple Vector class for physics calculations
class Vector {
  x: number;
  y: number;
  constructor(x = 0, y = 0) { this.x = x; this.y = y; }
  add(v: Vector) { this.x += v.x; this.y += v.y; return this; }
  limit(max: number) {
    const magSq = this.x * this.x + this.y * this.y;
    if (magSq > max * max) {
      const mag = Math.sqrt(magSq);
      this.x = (this.x / mag) * max;
      this.y = (this.y / mag) * max;
    }
    return this;
  }
}

class Particle {
  pos: Vector;
  vel: Vector;
  acc: Vector;
  lifespan: number;
  color: string;
  size: number;
  maxSpeed: number;

  constructor(x: number, y: number) {
    this.pos = new Vector(x, y);
    this.vel = new Vector(Math.random() - 0.5, Math.random() - 0.5);
    this.acc = new Vector(0, 0);
    this.lifespan = 255;
    this.color = Math.random() > 0.2 ? 'hsl(185, 80%, 50%)' : 'hsl(40, 100%, 55%)';
    this.size = Math.random() * 2 + 1;
    this.maxSpeed = Math.random() * 2 + 2;
  }

  applyForce(force: Vector) { this.acc.add(force); }

  update() {
    this.vel.add(this.acc);
    this.vel.limit(this.maxSpeed);
    this.pos.add(this.vel);
    this.acc = new Vector(0, 0);
    this.lifespan -= 1.5;
  }

  isDead() { return this.lifespan < 0; }
}

// --- MODIFIED: Audio Player Helper Function to return the Audio object ---
const playSound = (src: string, volume = 0.5, loop = false): HTMLAudioElement | null => {
    try {
        const audio = new Audio(src);
        audio.volume = volume;
        audio.loop = loop; // Set the loop property
        // Modern browsers require a user interaction to play audio.
        // Since this is called from an onClick, it's safe.
        audio.play().catch(e => console.error("Audio play failed:", e));
        return audio;
    } catch (e) {
        console.error("Audio initialization failed:", e);
        return null;
    }
};

export default function OracleLoginPage() {
  const [isSleeping, setIsSleeping] = useState(true);
  const [isFormVisible, setIsFormVisible] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);

  const starfieldCanvasRef = useRef<HTMLCanvasElement>(null);
  const oracleCanvasRef = useRef<HTMLCanvasElement>(null);
  const particleCanvasRef = useRef<HTMLCanvasElement>(null);

  const mousePosition = useRef({ x: 0, y: 0 });
  const animationFrameId = useRef<number>();
  const particles = useRef<Particle[]>([]);
  const humSoundRef = useRef<HTMLAudioElement | null>(null); // Ref to hold the looping audio

  // --- MODIFIED: Start looping sound ---
  const handleWakeSystem = () => {
    if (isSleeping) {
      // Start the looping hum sound and store its reference
      humSoundRef.current = playSound('/sounds/energy-hum-29083.mp3', 0.4, true);
      setIsFadingOut(true);
      setTimeout(() => setIsSleeping(false), 500);
    }
  };

  // --- MODIFIED: Stop looping sound and play another ---
  const handleOracleClick = () => {
    if (!isSleeping && !isFormVisible) {
      // The ambient hum continues to play.
      // We only play the one-shot scan sound.
      playSound('/sounds/deepscanmp3-14662.mp3', 0.6);
      setIsFormVisible(true);
    }
  };
  
  // Cleanup audio on component unmount
  useEffect(() => {
    return () => {
      if (humSoundRef.current) {
        // Fade out the sound for a smoother transition
        let vol = humSoundRef.current.volume;
        const fadeOutInterval = setInterval(() => {
          if (vol > 0.1) {
            vol -= 0.1;
            // Check if ref still exists before setting volume
            if(humSoundRef.current) {
                humSoundRef.current.volume = vol;
            }
          } else {
            if(humSoundRef.current) {
                humSoundRef.current.pause();
            }
            clearInterval(fadeOutInterval);
          }
        }, 50);
      }
    };
  }, []);

  const runAnimation = useCallback(() => {
    const s_canvas = starfieldCanvasRef.current;
    const o_canvas = oracleCanvasRef.current;
    const p_canvas = particleCanvasRef.current;
    if (!s_canvas || !o_canvas || !p_canvas) return;

    const s_ctx = s_canvas.getContext('2d');
    const o_ctx = o_canvas.getContext('2d');
    const p_ctx = p_canvas.getContext('2d');
    if (!s_ctx || !o_ctx || !p_ctx) return;

    const setupCanvases = () => {
      s_canvas.width = window.innerWidth;
      s_canvas.height = window.innerHeight;
      o_canvas.width = 500;
      o_canvas.height = 500;
      p_canvas.width = window.innerWidth;
      p_canvas.height = window.innerHeight;
    }
    setupCanvases();
    window.addEventListener('resize', setupCanvases);

    if (particles.current.length === 0) {
        for (let i = 0; i < 300; i++) {
            particles.current.push(new Particle(Math.random() * p_canvas.width, Math.random() * p_canvas.height));
        }
    }

    const animate = () => {
      s_ctx.clearRect(0, 0, s_canvas.width, s_canvas.height);
      s_ctx.fillStyle = "hsl(var(--background))";
      s_ctx.fillRect(0, 0, s_canvas.width, s_canvas.height);
      
      const stars = []; // Assuming starfield logic needs to be added back if desired.
      
      o_ctx.clearRect(0, 0, o_canvas.width, o_canvas.height);
      const o_centerX = o_canvas.width/2, o_centerY = o_canvas.height/2, radius = 200;
      const dx_eye = mousePosition.current.x - (isFormVisible ? (window.innerWidth/4) : (window.innerWidth/2)), dy_eye = mousePosition.current.y-(window.innerHeight/2);
      const angle_eye = Math.atan2(dy_eye, dx_eye), maxPupilDist = 40;
      const pupilX = o_centerX + Math.cos(angle_eye) * maxPupilDist, pupilY = o_centerY + Math.sin(angle_eye) * maxPupilDist;
      for (let i=0; i<60; i++) {
        o_ctx.beginPath(); o_ctx.strokeStyle=`hsla(185, 80%, 50%, ${0.05 + (i/60)*0.2})`; const x_offset = Math.sin(Date.now()*0.0005+i*0.1)*25; o_ctx.ellipse(o_centerX+x_offset, o_centerY, radius-i*3, radius, Math.PI/2, 0, Math.PI*2); o_ctx.stroke();
      }
      o_ctx.fillStyle = '#000'; o_ctx.beginPath(); o_ctx.arc(pupilX, pupilY, 40, 0, Math.PI*2); o_ctx.fill();
      o_ctx.fillStyle = 'hsl(var(--primary) / 0.5)'; o_ctx.beginPath(); o_ctx.arc(pupilX, pupilY, 15, 0, Math.PI*2); o_ctx.fill();
      o_ctx.fillStyle = 'hsl(var(--secondary) / 0.7)'; o_ctx.beginPath(); o_ctx.arc(pupilX, pupilY, 5, 0, Math.PI*2); o_ctx.fill();
      
      const attractor_x = isFormVisible ? (p_canvas.width / 4) : (p_canvas.width / 2);
      const attractor = new Vector(attractor_x, p_canvas.height / 2);
      p_ctx.clearRect(0, 0, p_canvas.width, p_canvas.height);
      for (let i = particles.current.length - 1; i >= 0; i--) {
        const p=particles.current[i]; const force=new Vector(attractor.x-p.pos.x, attractor.y-p.pos.y); const dist=Math.sqrt(force.x*force.x+force.y*force.y); const orbitalForce=new Vector(-force.y, force.x); orbitalForce.limit(0.1); force.limit(0.5); p.applyForce(force); if (dist>50) p.applyForce(orbitalForce); p.update(); p_ctx.beginPath(); p_ctx.fillStyle=p.color; p_ctx.globalAlpha=p.lifespan/255; p_ctx.arc(p.pos.x, p.pos.y, p.size, 0, Math.PI*2); p_ctx.fill(); p_ctx.globalAlpha=1.0; if(p.isDead()){particles.current[i]=new Particle(Math.random()*p_canvas.width,Math.random()*p_canvas.height);}
      }
      animationFrameId.current = requestAnimationFrame(animate);
    };
    animate();

    return () => { window.removeEventListener('resize', setupCanvases); }
  }, [isFormVisible]);

  useEffect(() => {
    if (!isSleeping) {
        const cleanup = runAnimation();
        const handleMouseMove = (e: MouseEvent) => { mousePosition.current = { x: e.clientX, y: e.clientY }; };
        window.addEventListener('mousemove', handleMouseMove);
        return () => {
            if (animationFrameId.current) { cancelAnimationFrame(animationFrameId.current); }
            window.removeEventListener('mousemove', handleMouseMove);
            if (cleanup) cleanup();
        };
    }
  }, [isSleeping, runAnimation]);

  return (
    <>
      <div className="oracle-container">
        {isSleeping && (
          <div id="oracle-sleep-screen" className={`sleep-screen ${isFadingOut ? 'fade-out' : ''}`} onClick={handleWakeSystem}>
            <div className="sleep-box">
              <h1 className="glitch vigilance-title" data-text="VigilanceAI">VigilanceAI</h1>
              <h2 className='sub-title'>The Ultimate Security System</h2>
              <p className="prompt-text">Click to Awaken</p>
            </div>
          </div>
        )}

        <canvas ref={starfieldCanvasRef} id="starfield-canvas" />
        <canvas ref={particleCanvasRef} id="particle-canvas" />

        {!isSleeping && (
            <div className={`auth-interface ${isFormVisible ? 'show-form' : ''}`}>
                <div className="eye-panel" onClick={handleOracleClick}>
                    <canvas ref={oracleCanvasRef} id="oracle-canvas" />
                </div>
                <div className="form-panel">
                    <LockScreenForm />
                </div>
            </div>
        )}
      </div>

      <style jsx global>{`
        body, html { margin: 0; padding: 0; overflow: hidden; background-color: hsl(var(--background)); }
        .oracle-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; }
        #starfield-canvas, #particle-canvas { position: fixed; top: 0; left: 0; z-index: 1; pointer-events: none; }
        
        .sleep-screen { z-index: 10; width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; background: hsl(var(--background)); cursor: pointer; opacity: 1; transition: opacity 0.5s ease-out; }
        .sleep-screen.fade-out { opacity: 0; pointer-events: none; }
        .sleep-box { padding: 40px 60px; border: 2px solid hsl(var(--secondary) / 0.5); box-shadow: 0 0 20px hsl(var(--secondary) / 0.5), inset 0 0 15px hsl(var(--secondary) / 0.4); text-align: center; }
        .vigilance-title { color: hsl(var(--secondary)); font-size: 3.5rem; text-shadow: 0 0 8px hsl(var(--secondary)); letter-spacing: 2px; margin: 0; font-family: 'Space Grotesk', monospace; }
        .sub-title { color: hsl(var(--foreground)); font-family: "Fira Code", monospace; font-size: 1.25rem; margin-top: 0.5rem; opacity: 0.8; }
        .prompt-text { color: hsl(var(--primary)); font-size: 1rem; margin-top: 2rem; letter-spacing: 3px; text-transform: uppercase; animation: pulse-text 2s infinite ease-in-out; }
        @keyframes pulse-text { 0%, 100% { opacity: 0.7; } 50% { opacity: 1; } }
        
        .glitch { position: relative; }
        .glitch:before, .glitch:after { content: attr(data-text); position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: hsl(var(--background)); overflow: hidden; }
        .glitch:before { left: 3px; text-shadow: -3px 0 hsl(var(--primary)); animation: glitch-anim-1 2.5s infinite linear alternate-reverse; }
        .glitch:after { left: -3px; text-shadow: -3px 0 hsl(var(--secondary)), 3px 3px hsl(var(--primary)); animation: glitch-anim-2 1.5s infinite linear alternate-reverse; }
        @keyframes glitch-anim-1 { 0% { clip-path: polygon(0 15%, 100% 15%, 100% 30%, 0 30%); } 100% { clip-path: polygon(0 70%, 100% 70%, 100% 85%, 0 85%); } }
        @keyframes glitch-anim-2 { 0% { clip-path: polygon(0 45%, 100% 45%, 100% 60%, 0 60%); } 100% { clip-path: polygon(0 5%, 100% 5%, 100% 20%, 0 20%); } }

        .auth-interface { position: relative; z-index: 2; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
        .eye-panel { flex: 1; display: flex; justify-content: center; align-items: center; cursor: pointer; transition: transform 0.8s ease-in-out; }
        .form-panel { flex: 0; width: 0; opacity: 0; transition: flex 0.8s ease-in-out, width 0.8s ease-in-out, opacity 0.5s ease-in-out; overflow: hidden; display: flex; justify-content: center; align-items: center; }
        .auth-interface.show-form .eye-panel { transform: translateX(-50%); }
        .auth-interface.show-form .form-panel { flex: 1; width: 50%; opacity: 1; }
        #oracle-canvas { width: 100%; max-width: 500px; height: auto; filter: drop-shadow(0 0 25px hsl(var(--secondary))); }
        
        .form-panel .bg-card { background: transparent !important; border: none !important; box-shadow: none !important; }
        .form-panel .text-primary { color: hsl(var(--primary)) !important; }
      `}</style>
    </>
  );
}