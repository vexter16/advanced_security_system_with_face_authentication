'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardFooter, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label"; 
import { Brain, Loader2, PlayCircle, HelpCircle, MessageSquare, AlertTriangle, ShieldAlert } from 'lucide-react';
import { useToast } from "@/hooks/use-toast";
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';

const PYTHON_FLASK_BASE_URL = process.env.NEXT_PUBLIC_PYTHON_FLASK_BASE_URL || "http://localhost:9003";

const videoFeedsConfig = [
    { id: 'cam1', name: 'CONTROL CENTER (RESTRICTED)', videoSrcRelative: 'videos/video7.mp4', type: 'restricted_zone' as const, posterUrl: `https://placehold.co/600x400/4B0082/FFFFFF?text=CONTROL+CENTER` },
    { id: 'cam2', name: 'PERIMETER FENCE - B', videoSrcRelative: 'videos/video2.mp4', type: 'standard' as const, posterUrl: `https://placehold.co/600x400/12222F/73A5C6?text=PERIMETER+B` },
    { id: 'cam3', name: 'LOGISTICS BAY 7', videoSrcRelative: 'videos/video3.mp4', type: 'standard' as const, posterUrl: `https://placehold.co/600x400/12222F/73A5C6?text=LOGISTICS+7` },
    { id: 'cam4', name: 'MESS HALL COURTYARD', videoSrcRelative: 'videos/video4.mp4', type: 'standard' as const, posterUrl: `https://placehold.co/600x400/12222F/73A5C6?text=MESS+HALL` },
    { id: 'cam5', name: 'COMMAND POST OVERWATCH', videoSrcRelative: 'videos/video1.mp4', type: 'standard' as const, posterUrl: `https://placehold.co/600x400/12222F/73A5C6?text=COMMAND+POST` },
];

type ThreatLevel = 'low' | 'medium' | 'high' | 'idle';
type CameraType = 'standard' | 'restricted_zone';
interface QwenAnalysisData {
  Detected_Activity?: string;
  Security_Status?: string;
  [key: string]: any;
}



interface FeedState {
  id: string; name: string; videoUrl: string; videoPathForApi: string; posterUrl: string;
  type: CameraType;
  parsedAnalysis?: QwenAnalysisData;
  streamingText: string;
  restrictedZoneResult?: RestrictedZoneResult;
  isAnalyzing: boolean;
  threatLevel: ThreatLevel;
  vqaQuestion: string;
  vqaAnswer: string;
  isVqaAnswering: boolean;
}

const initialFeedStates: FeedState[] = videoFeedsConfig.map(config => ({
  ...config,
  videoUrl: `${PYTHON_FLASK_BASE_URL}/${config.videoSrcRelative}`,
  videoPathForApi: config.videoSrcRelative,
  isAnalyzing: false,
  streamingText: '',
  threatLevel: 'idle',
  vqaQuestion: '',
  vqaAnswer: '',
  isVqaAnswering: false,
}));

const determineThreatLevel = (analysis?: QwenAnalysisData): ThreatLevel => {
    if (!analysis) return 'low';
    const status = (analysis.Security_Status || "").toLowerCase();
    const activity = (analysis.Detected_Activity || "").toLowerCase();

    if (status.includes("high threat") || activity.includes("explosion") || activity.includes("weapon")) {
        return "high";
    }
    if (status.includes("suspicious") || status.includes("potential threat") || activity.includes("breach")) {
        return "medium";
    }
    return "low";
};
const extractJson = (text: string): object | null => {
    // Find the first '{' and the last '}' to extract the JSON block
    const firstBrace = text.indexOf('{');
    const lastBrace = text.lastIndexOf('}');
    if (firstBrace === -1 || lastBrace === -1 || lastBrace < firstBrace) {
        return null;
    }
    const jsonString = text.substring(firstBrace, lastBrace + 1);
    try {
        return JSON.parse(jsonString);
    } catch (e) {
        console.error("Failed to parse extracted JSON:", e);
        return null;
    }
}

export default function LiveFeedDashboard() {
  const [feeds, setFeeds] = useState<FeedState[]>(initialFeedStates);
  const { toast } = useToast();

  const handleQwenAnalysis = async (feedId: string) => {
    const feed = feeds.find(f => f.id === feedId);
    if (!feed) return;

    setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, isAnalyzing: true, analysisResult: undefined, streamingText: '', threatLevel: 'idle' } : f));
    
    try {
      const response = await fetch('/api/analyze-qwen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: feed.videoPathForApi }),
      });
      if (!response.ok || !response.body) {
        const err = await response.json().catch(() => ({ error: "Analysis failed."}));
        throw new Error(err.error);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedJson = '';
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        accumulatedJson += chunk;
        setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, streamingText: accumulatedJson } : f));
      }

      const parsedJson = extractJson(accumulatedJson) as QwenAnalysisData | null;

            if (parsedJson) {
                const level = determineThreatLevel(parsedJson);
                setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, parsedAnalysis: parsedJson, threatLevel: level } : f));

                if (level === 'high') {
                    const alertMessage = `VigilanceAI HIGH THREAT: '${parsedJson?.Detected_Activity || 'Unspecified Threat'}' at feed '${feed.name}'.`;
                    fetch('/api/send-alert', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: alertMessage }), });
                    toast({ variant: 'destructive', duration: 10000, title: <div className="flex items-center font-bold"><AlertTriangle className="mr-2 h-6 w-6" /> HIGH THREAT DETECTED</div>, description: `Feed: ${feed.name}. Commander notified.` });
                } else {
                    toast({ title: `Analysis Complete: ${feed.name}` });
                }
            } else {
                toast({ variant: 'destructive', title: 'Data Error', description: 'Could not parse analysis from the AI response.' });
                // Keep the raw text visible for debugging
            }

        } catch (error: any) {
            toast({ variant: "destructive", title: `Analysis Failed: ${feed.name}`, description: error.message });
            setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, threatLevel: 'idle' } : f)); // Reset threat level on error
        } finally {
            setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, isAnalyzing: false } : f));
        }
    };
  
  const handleRestrictedZoneAnalysis = async (feedId: string) => {
        const feed = feeds.find(f => f.id === feedId);
        if (!feed) return;

        setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, isAnalyzing: true, restrictedZoneResult: undefined, threatLevel: 'idle' } : f));
        
        try {
            const response = await fetch('/api/analyze-restricted-zone', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_path: feed.videoPathForApi }),
            });
            const result: RestrictedZoneResult = await response.json();
            
            if (!response.ok) {
                throw new Error((result as any).error || "Failed to analyze restricted zone.");
            }

            setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, restrictedZoneResult: result, threatLevel: result.threatLevel } : f));

            if (result.threatLevel === 'high' || result.threatLevel === 'medium') {
                const alertMessage = `VigilanceAI ALERT [${result.threatLevel.toUpperCase()}]: ${result.summary}`;
                if (result.threatLevel === 'high') {
                  fetch('/api/send-alert', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: alertMessage }), });
                }
                toast({
                    variant: result.threatLevel === 'high' ? 'destructive' : 'default',
                    duration: 10000,
                    title: <div className="flex items-center font-bold"><ShieldAlert className="mr-2 h-6 w-6" /> ACCESS ALERT: {feed.name}</div>,
                    description: result.summary,
                });
            } else {
                toast({ title: `Zone Scan Complete: ${feed.name}`, description: result.summary });
            }
        } catch (error: any) {
            toast({ variant: "destructive", title: `Zone Scan Failed: ${feed.name}`, description: error.message });
        } finally {
            setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, isAnalyzing: false } : f));
        }
    };  

  const handleVqaQuestionChange = (feedId: string, question: string) => {
    setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, vqaQuestion: question } : f));
  };

  const handleAskVqaQuestion = async (feedId: string) => {
    const feed = feeds.find(f => f.id === feedId);
    if (!feed || !feed.vqaQuestion) return;
    setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, isVqaAnswering: true, vqaAnswer: '' } : f));
    try {
      const response = await fetch('/api/ask-question', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ video_path: feed.videoPathForApi, question: feed.vqaQuestion }), });
      if (!response.ok || !response.body) { const err = await response.json().catch(() => ({ error: "VQA failed."})); throw new Error(err.error); }
      const reader = response.body.getReader(); const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, vqaAnswer: f.vqaAnswer + chunk } : f));
      }
    } catch (error: any) { setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, vqaAnswer: `Error: ${error.message}` } : f)); }
    finally { setFeeds(prev => prev.map(f => f.id === feedId ? { ...f, isVqaAnswering: false } : f)); }
  };

  const getThreatGlowClass = (level: ThreatLevel) => {
    switch (level) {
      case 'high': return 'border-destructive shadow-destructive/40 hover:shadow-destructive/60';
      case 'medium': return 'border-primary shadow-primary/40 hover:shadow-primary/60';
      case 'low': return 'border-green-500 shadow-green-500/40 hover:shadow-green-500/60';
      default: return 'border-border hover:border-secondary';
    }
  };

  const renderQwenAnalysis = (feed: FeedState) => {
    if (feed.parsedAnalysis) {
            return Object.entries(feed.parsedAnalysis).map(([key, value]) => (
                value && <p key={key}><strong className="text-accent">{key.replace(/_/g, ' ')}:</strong> <span className="text-foreground/80 whitespace-pre-wrap">{Array.isArray(value) ? value.join(', ') : String(value)}</span></p>
            ));
        }
    if (feed.streamingText) {
        return <p className="whitespace-pre-wrap">{feed.streamingText}</p>;
    }
    if (feed.isAnalyzing) {
        return <p className="italic text-muted-foreground">Awaiting analysis stream...</p>;
    }
    return <p className="italic text-muted-foreground">Click button for automated analysis.</p>;
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-headline font-semibold text-primary text-center mb-8">Live Surveillance Network</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {feeds.map(feed => (
          <Card 
            key={feed.id} 
            className={cn("w-full overflow-hidden shadow-lg flex flex-col bg-card transition-all duration-500", getThreatGlowClass(feed.threatLevel))}
          >
            <CardHeader className="p-4">
              <CardTitle className="text-md font-headline flex items-center text-amber-200"><PlayCircle className="w-5 h-5 mr-2 text-accent" />{feed.name}</CardTitle>
            </CardHeader>
            <CardContent className="p-4 pt-0 space-y-3 flex-grow flex flex-col">
              <div className="aspect-video bg-muted rounded overflow-hidden relative border border-slate-700">
                <video src={feed.videoUrl} poster={feed.posterUrl} controls muted loop className="w-full h-full object-cover" />
              </div>
              {feed.type === 'restricted_zone' ? (
                                <div className='p-3 border-2 border-dashed border-destructive/50 rounded-lg space-y-3'>
                                    <h3 className='text-sm font-semibold flex items-center text-destructive'><ShieldAlert className='w-4 h-4 mr-2'/>Restricted Zone Access Control</h3>
                                    <Button onClick={() => handleRestrictedZoneAnalysis(feed.id)} disabled={feed.isAnalyzing || feed.isVqaAnswering} className="w-full bg-destructive hover:bg-destructive/90 text-destructive-foreground" size="sm">
                                        {feed.isAnalyzing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />SCANNING PERSONNEL...</> : 'Scan for Unauthorized Personnel'}
                                    </Button>
                                    <div className="qwen-output-preview text-xs p-3 border rounded-md min-h-[120px] ...">
                                        {feed.isAnalyzing && <p className="italic">Analyzing personnel...</p>}
                                        {feed.restrictedZoneResult && <p className='whitespace-pre-wrap'>{feed.restrictedZoneResult.summary}</p>}
                                        {!feed.isAnalyzing && !feed.restrictedZoneResult && <p className="italic">Click button to scan personnel in the zone.</p>}
                                    </div>
                                </div>
              ) : (
              <div className='p-3 border border-dashed rounded-lg space-y-3'>
                <h3 className='text-sm font-semibold flex items-center'><Brain className='w-4 h-4 mr-2'/>Automated Analysis</h3>
                <Button onClick={() => handleQwenAnalysis(feed.id)} disabled={feed.isAnalyzing || feed.isVqaAnswering} className="w-full bg-primary btn-primary-glow" size="sm">
                    {feed.isAnalyzing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />ANALYZING...</> : 'Run Automated Report'}
                </Button>
                <div className="qwen-output-preview text-xs p-3 border rounded-md min-h-[120px] overflow-y-auto bg-muted/50 space-y-2">
                    {renderQwenAnalysis(feed)}
                </div>
              </div>
              )}
            </CardContent>
            <CardFooter className="p-4 pt-0 flex flex-col items-start gap-3 bg-card-foreground/5">
              <Separator />
              <h3 className='text-sm font-semibold flex items-center w-full'><HelpCircle className='w-4 h-4 mr-2'/>Visual Question & Answer</h3>
              <div className='w-full space-y-2'>
                <Label htmlFor={`vqa-q-${feed.id}`} className='sr-only'>Question</Label>
                <Input id={`vqa-q-${feed.id}`} placeholder="e.g., Is there a green truck?" value={feed.vqaQuestion} onChange={(e) => handleVqaQuestionChange(feed.id, e.target.value)} disabled={feed.isVqaAnswering || feed.isAnalyzing} />
                <Button onClick={() => handleAskVqaQuestion(feed.id)} disabled={!feed.vqaQuestion || feed.isVqaAnswering || feed.isAnalyzing} className="w-full btn-glow" variant="secondary" size="sm">
                    {feed.isVqaAnswering ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Asking Qwen...</> : 'Ask Question'}
                </Button>
              </div>
              {(feed.vqaAnswer || feed.isVqaAnswering) && (
                <div className='w-full text-xs p-3 border rounded-md min-h-[80px] overflow-y-auto bg-muted/50'>
                    <h4 className='font-bold flex items-center mb-2'><MessageSquare className='w-4 h-4 mr-2'/>Answer:</h4>
                    {feed.isVqaAnswering && !feed.vqaAnswer && <p className='italic'>Waiting for answer from Qwen...</p>}
                    <p className='whitespace-pre-wrap'>{feed.vqaAnswer}</p>
                </div>
              )}
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}