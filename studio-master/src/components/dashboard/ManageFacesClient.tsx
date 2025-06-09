"use client";

import { useState, useEffect, useRef, useTransition } from 'react';
import type { FaceRecord } from '@/types';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from '@/components/ui/dialog';
import { PlusCircle, Trash2, Camera, UserX, Loader2, CheckCircle } from 'lucide-react';
import Image from 'next/image';
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';

const PYTHON_API_BASE_URL = process.env.NEXT_PUBLIC_PYTHON_FLASK_BASE_URL || "http://localhost:9003";

// This defines the multi-step capture process
const CAPTURE_STEPS = [
  { angle: "front", instruction: "Look straight at the camera (1/3)" },
  { angle: "front", instruction: "Look straight at the camera (2/3)" },
  { angle: "front", instruction: "Look straight at the camera (3/3)" },
  { angle: "left", instruction: "Slowly turn your head to your RIGHT (1/2)" },
  { angle: "left", instruction: "Slowly turn your head to your RIGHT (2/2)" },
  { angle: "right", instruction: "Slowly turn your head to your LEFT (1/2)" },
  { angle: "right", instruction: "Slowly turn your head to your LEFT (2/2)" },
];
const CAPTURE_DELAY_MS = 2000;

export default function ManageFacesClient() {
  const [faces, setFaces] = useState<FaceRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isPending, startTransition] = useTransition();
  const { toast } = useToast();
  const [newFaceRole, setNewFaceRole] = useState("Guard"); 
  const [newFaceName, setNewFaceName] = useState("");
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [currentCaptureStepIndex, setCurrentCaptureStepIndex] = useState(0);
  const [captureInstruction, setCaptureInstruction] = useState("");
  const [isCapturingSequence, setIsCapturingSequence] = useState(false);
  
  // These states are no longer needed for preview, but are essential for submission
  const [capturedFrontImages, setCapturedFrontImages] = useState<string[]>([]);
  const [capturedLeftImages, setCapturedLeftImages] = useState<string[]>([]);
  const [capturedRightImages, setCapturedRightImages] = useState<string[]>([]);
  const [newFacePassword, setNewFacePassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  // We need to pass the captured images to the submission function,
  // so we'll store them in a ref to avoid stale state issues in the callback.
  const submissionDataRef = useRef({
    name: "",
    front: [] as string[],
    left: [] as string[],
    right: [] as string[],
  });

  async function loadFaces() {
    setIsLoading(true);
    try {
      const response = await fetch(`${PYTHON_API_BASE_URL}/api/faces`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({message: "Failed to load faces"}));
        throw new Error(errorData.message || `Failed to load faces: ${response.statusText}`);
      }
      const fetchedFaces: FaceRecord[] = await response.json();
      setFaces(fetchedFaces.map(face => ({...face, addedDate: new Date(face.addedDate)})));
    } catch (error: any) {
      console.error("Error loading faces:", error);
      toast({ variant: "destructive", title: "Error Loading Faces", description: error.message });
      setFaces([]);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadFaces();
  }, []);

  const initializeCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast({ variant: 'destructive', title: 'Camera Not Supported' });
      setHasCameraPermission(false);
      return false;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setHasCameraPermission(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      return true;
    } catch (error) {
      console.error('Error accessing camera:', error);
      setHasCameraPermission(false);
      toast({ variant: 'destructive', title: 'Camera Access Denied', description: 'Please enable camera permissions.' });
      return false;
    }
  };

  useEffect(() => {
    if (isAddDialogOpen && hasCameraPermission === null) {
      initializeCamera();
    }
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
    };
  }, [isAddDialogOpen]);

  const captureImage = (): string | null => {
    if (videoRef.current && canvasRef.current && hasCameraPermission) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg');
      }
    }
    return null;
  };

  const startCaptureProcess = async () => {
    if (newFacePassword !== confirmPassword) {
      toast({ title: "Passwords do not match!", variant: "destructive" });
      return;
    }
    if (newFacePassword.length < 6) {
      toast({ title: "Password too short!", description: "Password must be at least 6 characters.", variant: "destructive" });
      return;
    }
    const cameraReady = await initializeCamera();
    if (!cameraReady) return;

    // Reset and start
    setIsCapturingSequence(true);
    setCurrentCaptureStepIndex(0);
    const tempFront: string[] = [];
    const tempLeft: string[] = [];
    const tempRight: string[] = [];

    // Update the ref with the current name
    submissionDataRef.current.name = newFaceName;

    const processNextCaptureStep = (stepIndex: number) => {
      // *** MODIFIED LOGIC: Trigger automatic submission ***
      if (stepIndex >= CAPTURE_STEPS.length) {
        setCaptureInstruction("All images captured! Saving data to database...");
        // Use the data collected during the sequence for submission
        submissionDataRef.current.front = tempFront;
        submissionDataRef.current.left = tempLeft;
        submissionDataRef.current.right = tempRight;
        handleAddFaceSubmit(); // Automatically call the submission function
        return;
      }

      const step = CAPTURE_STEPS[stepIndex];
      setCaptureInstruction(step.instruction);
      setCurrentCaptureStepIndex(stepIndex);

      setTimeout(() => {
        const imageDataUrl = captureImage();
        if (imageDataUrl) {
          if (step.angle === "front") tempFront.push(imageDataUrl);
          if (step.angle === "left") tempLeft.push(imageDataUrl);
          if (step.angle === "right") tempRight.push(imageDataUrl);
        } else {
          toast({ variant: "destructive", title: "Capture Failed", description: `Could not capture image for ${step.angle}. Sequence aborted.` });
          resetAddFaceForm(); // Reset on failure
          return;
        }
        processNextCaptureStep(stepIndex + 1);
      }, CAPTURE_DELAY_MS);
    };

    processNextCaptureStep(0);
  };

  const handleAddFaceSubmit = async () => {
    // Data is now taken from the ref to ensure it's up-to-date
    const { name, front, left, right } = submissionDataRef.current;
    if (newFacePassword !== confirmPassword) { toast({ title: "Passwords do not match!", variant: "destructive" }); return; }
    if (!newFaceName.trim() || !newFacePassword || !newFaceRole) { toast({ title: "Missing Fields", description: "Name, password, and role are all required.", variant: "destructive" }); return; }

    startTransition(async () => {
      try {
        const response = await fetch(`${PYTHON_API_BASE_URL}/api/faces`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: newFaceName,
            password: newFacePassword,
            role: newFaceRole,
            image_urls_front: front,
            image_urls_left: left,
            image_urls_right: right,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({message: "Failed to add face"}));
          throw new Error(errorData.message || `Failed to add face: ${response.statusText}`);
        }
        
        await loadFaces();
        setIsAddDialogOpen(false); // Close the dialog
        
        // *** MODIFIED LOGIC: Show prominent success toast ***
        toast({
            title: (
                <div className="flex items-center">
                    <CheckCircle className="w-6 h-6 mr-2 text-green-500" />
                    <span className="text-lg font-bold">Face Submitted Successfully</span>
                </div>
            ),
            description: `Operator '${name}' has been added to the secure database.`,
            duration: 5000, // Show for 5 seconds
        });
        // The reset function will be called by the onOpenChange handler of the Dialog
      } catch (error: any) {
        console.error("Error adding face:", error);
        toast({ title: "Error", description: error.message, variant: "destructive" });
        setIsCapturingSequence(false);
        setCaptureInstruction("Save failed. Please try again.");
      }
    });
  };
  
  const resetAddFaceForm = () => {
    setNewFaceName("");
    setHasCameraPermission(null);
    setIsCapturingSequence(false);
    setCaptureInstruction("");
    setCurrentCaptureStepIndex(0);
    // These are no longer needed for UI but good to reset
    setCapturedFrontImages([]);
    setCapturedLeftImages([]);
    setNewFaceRole("Guard"); // Reset role
    setCapturedRightImages([]);
    submissionDataRef.current = { name: "", front: [], left: [], right: [] };
    setNewFacePassword(""); // Reset password fields
    setConfirmPassword("");
  };

  const handleDeleteFace = async (id: string) => {
    startTransition(async () => {
      try {
        const response = await fetch(`${PYTHON_API_BASE_URL}/api/faces/${id}`, { method: 'DELETE' });
        if (!response.ok) {
           const errorData = await response.json().catch(() => ({message: "Failed to delete face"}));
          throw new Error(errorData.message || `Failed to delete face: ${response.statusText}`);
        }
        await loadFaces();
        toast({ title: "Success", description: "Face deleted successfully." });
      } catch (error: any) {
        console.error("Error deleting face:", error);
        toast({ title: "Error", description: error.message, variant: "destructive" });
      }
    });
  };

  return (
    <div>
      <Dialog open={isAddDialogOpen} onOpenChange={(isOpen) => {
        setIsAddDialogOpen(isOpen);
        if (!isOpen) resetAddFaceForm();
      }}>
        <DialogTrigger asChild>
          <Button size="lg" className="mb-6">
            <PlusCircle className="w-5 h-5 mr-2" /> Add New Face
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add New Authorized Operator</DialogTitle>
            <DialogDescription>
              Capture face images and set a password for this operator.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            {/* Name Input (same) */}
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">Name</Label>
              <Input id="name" value={newFaceName} onChange={(e) => setNewFaceName(e.target.value)} className="col-span-3" required disabled={isCapturingSequence || isPending} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="password" className="text-right">Password</Label>
              <Input id="password" type="password" value={newFacePassword} onChange={(e) => setNewFacePassword(e.target.value)} className="col-span-3" required disabled={isCapturingSequence || isPending} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="confirm-password" className="text-right">Confirm</Label>
              <Input id="confirm-password" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="col-span-3" required disabled={isCapturingSequence || isPending} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="role-select" className="text-right">Role</Label>
              <Select value={newFaceRole} onValueChange={setNewFaceRole} disabled={isCapturingSequence || isPending}>
                <SelectTrigger id="role-select" className="col-span-3">
                  <SelectValue placeholder="Select a role" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Commander">Commander</SelectItem>
                  <SelectItem value="Lead Analyst">Lead Analyst</SelectItem>
                  <SelectItem value="Operator">Operator</SelectItem>
                  <SelectItem value="Guard">Guard</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-4 text-center">
              {!isCapturingSequence && (
                <Button onClick={startCaptureProcess} disabled={isPending || !newFaceName.trim()}>
                  <Camera className="mr-2 h-4 w-4" /> Start Capture Sequence
                </Button>
              )}
              {isCapturingSequence && (
                <div className="flex items-center justify-center text-lg font-semibold text-primary">
                  <Loader2 className="w-6 h-6 mr-2 animate-spin" /> {captureInstruction}
                </div>
              )}
              {isPending && ( // Show saving status
                 <div className="flex items-center justify-center text-lg font-semibold text-green-600">
                   <Loader2 className="w-6 h-6 mr-2 animate-spin" /> Saving to database...
                 </div>
              )}
            </div>

            <div className="relative w-full aspect-video bg-muted rounded-md overflow-hidden border border-dashed">
              <video ref={videoRef} className="w-full h-full object-cover" autoPlay muted playsInline />
              <canvas ref={canvasRef} className="hidden"></canvas>
            </div>
          </div>

          <DialogFooter>
            {/* *** MODIFIED LOGIC: Only the Cancel button remains *** */}
            <DialogClose asChild>
              <Button type="button" variant="outline" disabled={isPending || isCapturingSequence}>Cancel</Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <h3 className="text-lg font-semibold mt-8 mb-4">Existing Faces</h3>
      {isLoading ? (
        <div className="text-center py-10"><Loader2 className="mx-auto h-10 w-10 animate-spin text-primary" /> <p className="mt-2 text-muted-foreground">Loading faces...</p></div>
      ) : faces.length === 0 ? (
         <div className="text-center py-10 text-muted-foreground border-2 border-dashed rounded-lg">
          <UserX className="mx-auto h-12 w-12" />
          <p className="mt-4 text-lg">No faces in the database.</p>
          <p>Click "Add New Face" to get started.</p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[80px]">Front</TableHead>
                <TableHead className="w-[80px]">Left</TableHead>
                <TableHead className="w-[80px]">Right</TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Added Date</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {faces.map((face) => (
                <TableRow key={face.id}>
                  <TableCell>
                    <div className="w-16 h-16 rounded-md overflow-hidden bg-muted flex items-center justify-center">
                      {face.imageUrlsFront?.[0] ? (
                        <Image src={face.imageUrlsFront[0]} alt={`${face.name} - Front`} width={64} height={64} className="object-cover" />
                      ) : <UserX className="w-8 h-8 text-muted-foreground" />}
                    </div>
                  </TableCell>
                   <TableCell>
                    <div className="w-16 h-16 rounded-md overflow-hidden bg-muted flex items-center justify-center">
                      {face.imageUrlsLeft?.[0] ? (
                        <Image src={face.imageUrlsLeft[0]} alt={`${face.name} - Left`} width={64} height={64} className="object-cover" />
                      ) : <UserX className="w-8 h-8 text-muted-foreground" />}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="w-16 h-16 rounded-md overflow-hidden bg-muted flex items-center justify-center">
                      {face.imageUrlsRight?.[0] ? (
                        <Image src={face.imageUrlsRight[0]} alt={`${face.name} - Right`} width={64} height={64} className="object-cover" />
                      ) : <UserX className="w-8 h-8 text-muted-foreground" />}
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">{face.name}</TableCell>
                  <TableCell>{new Date(face.addedDate).toLocaleDateString()}</TableCell>
                  <TableCell className="text-right">
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      onClick={() => handleDeleteFace(face.id)}
                      disabled={isPending}
                      aria-label={`Delete face for ${face.name}`}
                    >
                      <Trash2 className="w-4 h-4 text-destructive" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}