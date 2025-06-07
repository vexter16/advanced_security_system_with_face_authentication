
"use client";

import type { SurveillanceEvent } from "@/types";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button"; // Added Button
import { AlertTriangle, Clock, Brain, Loader2, PlayCircle } from 'lucide-react'; // Replaced Eye with PlayCircle or similar
import { assessThreatLevel, type ThreatAssessmentOutput } from "@/ai/flows/assess-threat-level";
import { describeScene } from "@/ai/flows/describe-scene";
import { useState } from "react";
import { useToast } from "@/hooks/use-toast";

interface ThreatAlertCardProps {
  event: SurveillanceEvent;
  onProcessComplete: (eventId: string, sceneDescription: string, threatAssessment: ThreatAssessmentOutput) => void;
}

const ThreatBadge: React.FC<{ level?: 'low' | 'medium' | 'high' }> = ({ level }) => {
  if (!level) return null;
  let variant: "default" | "secondary" | "destructive" | "outline" = "default";
  let text = level.toUpperCase();

  switch (level) {
    case 'low': variant = "secondary"; text = "LOW"; break;
    case 'medium': variant = "default"; text = "MEDIUM"; break;
    case 'high': variant = "destructive"; text = "HIGH"; break;
  }
  return <Badge variant={variant} className="font-bold">{text}</Badge>;
};

export default function ThreatAlertCard({ event, onProcessComplete }: ThreatAlertCardProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentThreatAssessment, setCurrentThreatAssessment] = useState<ThreatAssessmentOutput | undefined>(event.threatAssessment);
  const [currentSceneDescription, setCurrentSceneDescription] = useState<string | undefined>(event.sceneDescription);
  const { toast } = useToast();

  const handleQwenAnalysis = async () => {
    setIsProcessing(true);
    setCurrentSceneDescription(undefined); // Clear previous results
    setCurrentThreatAssessment(undefined); // Clear previous results
    try {
      // Use posterUrl for describeScene as it expects an image data URI or URL
      const sceneResult = await describeScene({ frameDataUri: event.posterUrl });
      setCurrentSceneDescription(sceneResult.sceneDescription);
      
      const threatResult = await assessThreatLevel({ sceneDescription: sceneResult.sceneDescription });
      setCurrentThreatAssessment(threatResult);
      
      onProcessComplete(event.id, sceneResult.sceneDescription, threatResult);
      toast({ title: "Analysis Complete", description: `Qwen analysis for ${event.name} finished.` });

    } catch (error) {
      console.error("Error processing event:", error);
      const errorDescription = "Error during AI analysis.";
      const errorAssessment: ThreatAssessmentOutput = { threatLevel: 'low', description: 'Error assessing threat due to analysis failure.' };
      
      setCurrentSceneDescription(errorDescription);
      setCurrentThreatAssessment(errorAssessment);
      onProcessComplete(event.id, errorDescription, errorAssessment);
      toast({ variant: "destructive", title: "Analysis Failed", description: "Could not complete Qwen analysis." });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Card className="w-full overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300 ease-in-out flex flex-col">
      <CardHeader className="p-4">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-lg font-headline flex items-center">
              <PlayCircle className="w-6 h-6 mr-2 text-primary" /> {event.name}
            </CardTitle>
            <CardDescription className="text-xs flex items-center mt-1">
              <Clock className="w-3 h-3 mr-1" /> Last update: {new Date(event.timestamp).toLocaleTimeString()}
            </CardDescription>
          </div>
          {currentThreatAssessment && !isProcessing && <ThreatBadge level={currentThreatAssessment.threatLevel} />}
          {isProcessing && (
             <Badge variant="outline" className="flex items-center">
              <Loader2 className="w-4 h-4 mr-1 animate-spin" /> Processing
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-4 space-y-3 flex-grow">
        <div className="aspect-video bg-muted rounded overflow-hidden relative">
          <video 
            src={event.videoUrl} 
            poster={event.posterUrl}
            controls 
            muted
            loop
            className="w-full h-full object-cover"
            data-ai-hint="security footage"
          >
            Your browser does not support the video tag.
          </video>
        </div>
        
        <Button 
          onClick={handleQwenAnalysis} 
          disabled={isProcessing}
          className="w-full"
          variant="outline"
        >
          {isProcessing ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing...
            </>
          ) : (
            <>
              <Brain className="w-4 h-4 mr-2" /> Qwen Analysis
            </>
          )}
        </Button>

        {(isProcessing && !currentSceneDescription) && (
           <div className="text-sm text-muted-foreground flex items-center pt-2">
             <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating scene description...
           </div>
        )}
        {currentSceneDescription && (
          <div>
            <h4 className="text-sm font-semibold mb-1 flex items-center pt-2"><Brain className="w-4 h-4 mr-2 text-accent" />AI Scene Analysis:</h4>
            <p className="text-xs text-muted-foreground leading-relaxed">{currentSceneDescription}</p>
          </div>
        )}
      </CardContent>
      {currentThreatAssessment && !isProcessing && (
        <CardFooter className="p-4 bg-card-foreground/5 mt-auto">
          <div>
            <h4 className="text-sm font-semibold mb-1 flex items-center"><AlertTriangle className="w-4 h-4 mr-2 text-destructive" />Threat Reasoning:</h4>
            <p className="text-xs text-muted-foreground">{currentThreatAssessment.description}</p>
          </div>
        </CardFooter>
      )}
    </Card>
  );
}
