
import type { ThreatAssessmentOutput } from '@/ai/flows/assess-threat-level';

export type DetectedObjectType = 'weapon' | 'soldier' | 'civilian' | 'vehicle';

export interface DetectedObject {
  id: string;
  type: DetectedObjectType;
  x: number;
  y: number;
}

export interface SurveillanceEvent {
  id: string;
  name: string; 
  timestamp: Date; 
  videoUrl: string; 
  posterUrl: string; 
  sceneDescription?: string;
  threatAssessment?: ThreatAssessmentOutput;
}

export interface FaceRecord {
  id: string;
  name: string;
  imageUrlsFront: string[]; // Array for multiple front images
  imageUrlsLeft?: string[];  // Array for multiple left profile images
  imageUrlsRight?: string[]; // Array for multiple right profile images
  addedDate: Date;
}
