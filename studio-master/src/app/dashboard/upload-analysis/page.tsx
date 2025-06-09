'use client';

import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, Loader2, CheckCircle, AlertTriangle, HelpCircle, MessageSquare } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Separator } from '@/components/ui/separator';

type AnalysisStatus = 'idle' | 'uploading' | 'analyzing' | 'complete' | 'error';

interface AnalysisResult {
  temp_filename: string;
  yolo_yamnet_summary: any;
  qwen_analysis_result: string;
}

export default function UploadAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [streamingText, setStreamingText] = useState('');
  const [error, setError] = useState<string | null>(null);
  
  // --- NEW VQA STATES ---
  const [vqaQuestion, setVqaQuestion] = useState('');
  const [vqaAnswer, setVqaAnswer] = useState('');
  const [isVqaAnswering, setIsVqaAnswering] = useState(false);

  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setStatus('idle');
      setResult(null);
      setError(null);
      setStreamingText('');
      setVqaQuestion('');
      setVqaAnswer('');
    }else{
      setSelectedFile(null);
      setVideoSrc(null);
    }
  };

  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setVideoSrc(url);

      // Cleanup function to revoke the object URL when the component unmounts
      // or the selectedFile changes, preventing memory leaks.
      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [selectedFile]);

  // --- NEW: Cleanup logic when user leaves the page ---
  useEffect(() => {
    return () => {
      if (result?.temp_filename) {
        navigator.sendBeacon('/api/cleanup-upload', JSON.stringify({ temp_filename: result.temp_filename }));
      }
    };
  }, [result]);

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setStatus('uploading'); setResult(null); setError(null); setStreamingText('');
    const formData = new FormData();
    formData.append('video', selectedFile);
    try {
        const response = await fetch('/api/upload-and-analyze', { method: 'POST', body: formData });
        if (!response.ok || !response.body) {
            const err = await response.json().catch(() => ({error: 'Analysis failed.'}));
            throw new Error(err.error || 'Unknown server error');
        }
        setStatus('analyzing'); toast({ title: "Upload Complete", description: "Server performing deep analysis..." });
        const reader = response.body.getReader(); const decoder = new TextDecoder();
        let finalJsonString = '';
        while(true) {
            const { value, done } = await reader.read();
            if (done) { try { const finalResult = JSON.parse(finalJsonString); setResult(finalResult); } catch (e) { setError("Failed to parse final result."); } break; }
            const chunk = decoder.decode(value, { stream: true });
            finalJsonString += chunk; setStreamingText(prev => prev + chunk);
        }
        setStatus('complete'); toast({ title: 'Deep Analysis Complete' });
    } catch (e: any) { setError(e.message); setStatus('error'); toast({ variant: 'destructive', title: 'Error', description: e.message }); }
  };

  const handleAskVqaQuestion = async () => {
    if (!vqaQuestion || !result?.temp_filename) {
      toast({ variant: 'destructive', title: 'Error', description: 'A question and an analyzed video are required.' });
      return;
    }
    setIsVqaAnswering(true);
    setVqaAnswer('');
    
    try {
        // *** THE FIX: Call the correct, existing API route ***
        const response = await fetch('/api/ask-question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Send temp_filename instead of video_path
            body: JSON.stringify({
                temp_filename: result.temp_filename,
                question: vqaQuestion
            }),
        });

        if (!response.ok || !response.body) {
            const err = await response.json().catch(() => ({ error: "VQA failed."}));
            throw new Error(err.error || 'Unknown VQA error');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            setVqaAnswer(prev => prev + chunk);
        }
    } catch (e: any) {
        setVqaAnswer(`Error: ${e.message}`);
        toast({ variant: "destructive", title: "Visual Q&A Failed", description: e.message });
    } finally {
        setIsVqaAnswering(false);
    }
  };

  const renderAnalysisResults = () => { /* ... (Unchanged) ... */
    if (status === 'error' && error) return <Card className="border-destructive"><CardHeader><CardTitle className="flex items-center text-destructive"><AlertTriangle className="mr-2"/>Error</CardTitle></CardHeader><CardContent><p>{error}</p></CardContent></Card>;
    if (status === 'analyzing' || status === 'complete') {
        if (result) return <Card className="bg-muted/30"><CardHeader><CardTitle className="flex items-center"><CheckCircle className="text-green-500 mr-2"/>Analysis Complete</CardTitle></CardHeader><CardContent className="space-y-4"><div><h3 className="font-bold text-lg mb-2">Qwen-VL Assessment</h3><p className="text-sm whitespace-pre-wrap p-3 bg-background rounded-md">{result.qwen_analysis_result}</p></div><div><h3 className="font-bold text-lg mb-2">YOLO/YAMNet Summary</h3><pre className="text-xs p-3 bg-background rounded-md overflow-x-auto">{JSON.stringify(result.yolo_yamnet_summary, null, 2)}</pre></div></CardContent></Card>;
        if (streamingText) return <Card className="bg-muted/30"><CardHeader><CardTitle className="flex items-center"><Loader2 className="mr-2 animate-spin"/>Receiving Analysis...</CardTitle></CardHeader><CardContent><pre className="text-xs p-3 bg-background rounded-md overflow-x-auto whitespace-pre-wrap">{streamingText}</pre></CardContent></Card>;
    }
    return null;
  }

  return (
    <div className="container mx-auto py-8">
      <Card className="max-w-4xl mx-auto">
        <CardHeader>
          <CardTitle className="text-2xl font-headline flex items-center"><Upload className="w-7 h-7 mr-3 text-primary" />Upload & Deep Analyze Video</CardTitle>
          <CardDescription>Upload a video to perform a full YOLO/YAMNet analysis, then ask specific questions about its content.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid w-full items-center gap-2">
            <Label htmlFor="video-upload">Video File</Label>
            <Input id="video-upload" type="file" accept="video/mp4,video/webm" onChange={handleFileChange} disabled={status === 'uploading' || status === 'analyzing'}/>
          </div>
          {videoSrc && (
            <div className="p-4 border rounded-md">
                <video key={videoSrc} controls className="w-full max-h-96 rounded">
                    <source src={videoSrc} type={selectedFile?.type} />
                    Your browser does not support the video tag.
                </video>
            </div>
          )}
          <Button onClick={handleAnalyze} disabled={!selectedFile || status === 'analyzing' || status === 'uploading'} className="w-full" size="lg">
            {(status === 'uploading' || status === 'analyzing') ? <><Loader2 className="mr-2 h-5 w-5 animate-spin" />{status === 'uploading' ? 'Uploading...' : 'Analyzing...'}</> : 'Run Deep Analysis'}
          </Button>

          {renderAnalysisResults()}

          {/* --- NEW VQA SECTION FOR UPLOAD PAGE --- */}
          {status === 'complete' && result && (
            <div className='pt-6'>
                <Separator />
                <div className="pt-6 space-y-4">
                    <h3 className='text-xl font-semibold flex items-center'><HelpCircle className='w-6 h-6 mr-2'/>Ask a Follow-up Question</h3>
                    <div className='w-full space-y-2'>
                        <Label htmlFor="vqa-upload-q">Your Question</Label>
                        <Input id="vqa-upload-q" placeholder="e.g., How many soldiers are wearing helmets?" value={vqaQuestion} onChange={(e) => setVqaQuestion(e.target.value)} disabled={isVqaAnswering}/>
                        <Button onClick={handleAskVqaQuestion} disabled={!vqaQuestion || isVqaAnswering} className="w-full" variant="secondary">
                            {isVqaAnswering ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Asking Qwen...</> : 'Ask Question'}
                        </Button>
                    </div>
                    {(vqaAnswer || isVqaAnswering) && (
                        <div className='w-full text-sm p-4 border rounded-md min-h-[100px] bg-muted/50'>
                            <h4 className='font-bold flex items-center mb-2'><MessageSquare className='w-4 h-4 mr-2'/>Answer:</h4>
                            {isVqaAnswering && !vqaAnswer && <p className='italic'>Waiting for answer from Qwen...</p>}
                            <p className='whitespace-pre-wrap'>{vqaAnswer}</p>
                        </div>
                    )}
                </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}