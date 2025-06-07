'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ShieldCheck, Camera, Loader2, KeyRound } from 'lucide-react';
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

const AUTH_API_BASE_URL = process.env.NEXT_PUBLIC_PYTHON_FLASK_BASE_URL || "http://localhost:9003";

// This new type defines the two possible views for our form
type AuthMode = 'face' | 'password';

export default function LockScreenForm() {
  const router = useRouter();
  const { toast } = useToast();
  
  // This state controls which view is shown: 'face' or 'password'
  const [authMode, setAuthMode] = useState<AuthMode>('face');
  
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [authMessage, setAuthMessage] = useState<{text: string, type: 'error' | 'success' | 'info'} | null>(null);

  // States for Face Recognition
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // States for Password Login
  const [operators, setOperators] = useState<string[]>([]);
  const [selectedOperator, setSelectedOperator] = useState('');
  const [password, setPassword] = useState('');

  // This effect runs when the user switches to 'password' mode
  useEffect(() => {
    // Only fetch if we are in password mode and haven't fetched the list yet
    if (authMode === 'password' && operators.length === 0) {
      setIsAuthenticating(true); // Show a loading state
      const fetchOperators = async () => {
        try {
          const res = await fetch(`${AUTH_API_BASE_URL}/api/operators`);
          if (!res.ok) throw new Error('Failed to fetch operator list from server.');
          const data = await res.json();
          setOperators(data);
          // Pre-select the first operator in the list
          if (data.length > 0) {
            setSelectedOperator(data[0]);
          }
        } catch (error: any) {
          toast({ variant: 'destructive', title: 'Network Error', description: error.message });
        } finally {
          setIsAuthenticating(false);
        }
      };
      fetchOperators();
    }
  }, [authMode, operators.length, toast]);

  // This effect manages the camera stream
  useEffect(() => {
    // Initialize camera only if we are in 'face' mode
    if (authMode === 'face' && hasCameraPermission === null) {
      const initializeCamera = async () => {
        if (!navigator.mediaDevices?.getUserMedia) return setHasCameraPermission(false);
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          if (videoRef.current) videoRef.current.srcObject = stream;
          setHasCameraPermission(true);
        } catch (error) {
          setHasCameraPermission(false);
        }
      };
      initializeCamera();
    }
    // Cleanup function to stop the camera stream when component unmounts or mode changes
    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [authMode, hasCameraPermission]);

  const captureFrame = (): string | null => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current; const canvas = canvasRef.current;
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.9);
      }
    }
    return null;
  };

  const handleFaceAuthenticate = async () => {
    const frameDataUrl = captureFrame();
    if (!frameDataUrl) return;
    setIsAuthenticating(true); setAuthMessage({text: "Verifying face...", type: 'info'});
    try {
      const response = await fetch(`${AUTH_API_BASE_URL}/api/authenticate-operator`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ image_data_url: frameDataUrl })});
      const result = await response.json();
      if (response.ok && result.authenticated) {
        setAuthMessage({text: result.message, type: 'success'});
        localStorage.setItem('isAuthenticatedVigilanceAI', 'true');
        toast({ title: "Authentication Successful", description: result.message });
        router.push('/dashboard');
      } else {
        setAuthMessage({text: result.message || "Face not recognized.", type: 'error'});
        toast({ variant: "destructive", title: "Authentication Failed", description: result.message || "Please try again or use password." });
      }
    } catch (error: any) {
      setAuthMessage({text: "Cannot connect to auth service.", type: 'error'});
    } finally {
      setIsAuthenticating(false);
    }
  };
  
  const handlePasswordAuthenticate = async (e: React.FormEvent) => {
    e.preventDefault(); // Prevent form submission from reloading the page
    if (!selectedOperator || !password) {
      setAuthMessage({ text: "Please select an operator and enter a password.", type: 'error' });
      return;
    }
    setIsAuthenticating(true); setAuthMessage({text: "Verifying credentials...", type: 'info'});
    try {
      const response = await fetch(`${AUTH_API_BASE_URL}/api/authenticate-password`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ name: selectedOperator, password })});
      const result = await response.json();
      if (response.ok && result.authenticated) {
        setAuthMessage({text: result.message, type: 'success'});
        localStorage.setItem('isAuthenticatedVigilanceAI', 'true');
        toast({ title: "Authentication Successful", description: result.message });
        router.push('/dashboard');
      } else {
        setAuthMessage({text: result.message || "Invalid credentials.", type: 'error'});
        toast({ variant: "destructive", title: "Authentication Failed", description: result.message });
      }
    } catch (error: any) {
      setAuthMessage({text: "Cannot connect to auth service.", type: 'error'});
    } finally {
      setIsAuthenticating(false);
    }
  };

  // This function renders the Face Recognition UI
  const renderFaceAuth = () => (
    <>
      <div className="w-64 h-48 bg-muted rounded-lg flex items-center justify-center overflow-hidden border-2 border-dashed border-border relative">
        <video ref={videoRef} className="w-full h-full object-cover" autoPlay muted playsInline style={{ display: hasCameraPermission ? 'block' : 'none' }} />
        <canvas ref={canvasRef} className="hidden" />
        {hasCameraPermission === null && <div className="absolute inset-0 flex flex-col items-center justify-center bg-card/80"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>}
        {hasCameraPermission === false && <div className="absolute inset-0 flex flex-col items-center justify-center p-4 text-center"><Camera className="w-10 h-10 text-destructive mb-2" /><p className="text-sm text-destructive-foreground">Camera access denied.</p></div>}
      </div>

      <Button onClick={handleFaceAuthenticate} disabled={isAuthenticating || !hasCameraPermission} className="w-full bg-primary hover:bg-primary/90 text-primary-foreground btn-primary-glow" size="lg">
        {isAuthenticating ? <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5" /> : <Camera className="mr-2 h-5 w-5" />}
        Authenticate Operator
      </Button>

      <Button variant="link" className="text-sm h-auto p-0" onClick={() => { setAuthMode('password'); setAuthMessage(null); }}>
        Use Password Instead
      </Button>
    </>
  );

  // This function renders the Password Login UI
  const renderPasswordAuth = () => (
    <form onSubmit={handlePasswordAuthenticate} className='w-full max-w-sm space-y-6'>
        <div className='space-y-2'>
            <Label htmlFor="operator-select">Operator Name</Label>
            <Select value={selectedOperator} onValueChange={setSelectedOperator} disabled={isAuthenticating}>
                <SelectTrigger id="operator-select" className='w-full input-glow'>
                    <SelectValue placeholder="Select Operator..." />
                </SelectTrigger>
                <SelectContent>
                    {operators.length > 0 ? (
                        operators.map(op => <SelectItem key={op} value={op}>{op}</SelectItem>)
                    ) : (
                        <SelectItem value="loading" disabled>Loading operators...</SelectItem>
                    )}
                </SelectContent>
            </Select>
        </div>
        <div className='space-y-2'>
            <Label htmlFor="password-input">Password</Label>
            <Input id="password-input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} disabled={isAuthenticating} className='input-glow' />
        </div>
      
        <Button type="submit" disabled={isAuthenticating || !selectedOperator} className="w-full bg-primary hover:bg-primary/90 text-primary-foreground btn-primary-glow" size="lg">
            {isAuthenticating ? <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5" /> : <KeyRound className="mr-2 h-5 w-5"/>}
            Authenticate
        </Button>

        <Button variant="link" className="text-sm h-auto p-0" onClick={() => { setAuthMode('face'); setAuthMessage(null); }}>
            Use Face Recognition Instead
        </Button>
    </form>
  );

  return (
    // The main return now just decides which of the two render functions to call
    // This keeps the JSX clean and easy to read.
    <Card className="w-full max-w-md shadow-2xl bg-card text-card-foreground border-secondary/50">
      <CardHeader className="text-center">
        <div className="inline-flex items-center justify-center mb-4">
          <ShieldCheck className="w-12 h-12 text-primary" />
        </div>
        <CardTitle className="text-3xl font-headline text-primary">VigilanceAI</CardTitle>
        <CardDescription className="text-muted-foreground">
          {authMode === 'face' ? 'Secure Access via Facial Recognition' : 'Operator Password Authentication'}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col items-center space-y-6 px-8 pb-8">
        {authMessage && (
          <Alert variant={authMessage.type === 'error' ? 'destructive' : 'default'} className="w-full">
            <AlertTitle>{authMessage.type.toUpperCase()}</AlertTitle>
            <AlertDescription>{authMessage.text}</AlertDescription>
          </Alert>
        )}
        
        {authMode === 'face' ? renderFaceAuth() : renderPasswordAuth()}
      </CardContent>
    </Card>
  );
}