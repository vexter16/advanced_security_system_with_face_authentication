// This is an AI-powered security automation system for military camps.
//
// The surveillance feeds are passed through YOLO and YAMNet models to detect objects.
// Any civilian detected is considered a threat.
// When a weapon detected is near a civilian(relative distance of weapon is short wrt civilian than a soldier) then it is a bigger threat.
// In either case the surveillance cam footage is then passed through a vision llm(qwen).
// Ofc frames are passed and realtime description of scene should be generated and stored.

'use server';

/**
 * @fileOverview Describes the scene in a surveillance footage frame.
 *
 * - describeScene - A function that takes a frame and returns a description of the scene.
 * - DescribeSceneInput - The input type for the describeScene function.
 * - DescribeSceneOutput - The return type for the describeScene function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const DescribeSceneInputSchema = z.object({
  frameDataUri: z
    .string()
    .describe(
      "A frame from the surveillance footage, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
});
export type DescribeSceneInput = z.infer<typeof DescribeSceneInputSchema>;

const DescribeSceneOutputSchema = z.object({
  sceneDescription: z
    .string()
    .describe('A description of the scene in the surveillance footage.'),
});
export type DescribeSceneOutput = z.infer<typeof DescribeSceneOutputSchema>;

export async function describeScene(input: DescribeSceneInput): Promise<DescribeSceneOutput> {
  return describeSceneFlow(input);
}

const describeScenePrompt = ai.definePrompt({
  name: 'describeScenePrompt',
  input: {schema: DescribeSceneInputSchema},
  output: {schema: DescribeSceneOutputSchema},
  prompt: `You are an AI that describes the scene in a surveillance footage frame.

  Here is the frame:
  {{media url=frameDataUri}}

  Describe the scene in detail, including the objects present, their locations, and any activities taking place.`,
});

const describeSceneFlow = ai.defineFlow(
  {
    name: 'describeSceneFlow',
    inputSchema: DescribeSceneInputSchema,
    outputSchema: DescribeSceneOutputSchema,
  },
  async input => {
    const {output} = await describeScenePrompt(input);
    return output!;
  }
);
