
'use server';

/**
 * @fileOverview Assesses the threat level based on a scene description.
 *
 * - assessThreatLevel - A function that assesses the threat level.
 * - ThreatAssessmentInput - The input type for the assessThreatLevel function.
 * - ThreatAssessmentOutput - The return type for the assessThreatLevel function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const ThreatAssessmentInputSchema = z.object({
  sceneDescription: z.string().describe('The detailed description of the scene to be assessed.'),
});
export type ThreatAssessmentInput = z.infer<typeof ThreatAssessmentInputSchema>;

const ThreatAssessmentOutputSchema = z.object({
  threatLevel: z.enum(['low', 'medium', 'high']).describe('The assessed threat level.'),
  description: z.string().describe('A description of the threat and reasoning behind the assessment.'),
});
export type ThreatAssessmentOutput = z.infer<typeof ThreatAssessmentOutputSchema>;

export async function assessThreatLevel(input: ThreatAssessmentInput): Promise<ThreatAssessmentOutput> {
  return assessThreatLevelFlow(input);
}

const prompt = ai.definePrompt({
  name: 'assessThreatLevelPrompt',
  input: {schema: ThreatAssessmentInputSchema},
  output: {schema: ThreatAssessmentOutputSchema},
  prompt: `You are a security expert assessing threat levels in a military camp based on a scene description.

  Analyze the following scene description:
  {{{sceneDescription}}}

  Based on this description, assess the threat level (low, medium, or high) and provide detailed reasoning for your assessment.
  Consider terms like 'weapon', 'explosion', 'unauthorized access', 'direct attack', 'fire' as indicators of high threat.
  Consider terms like 'suspicious individual', 'civilian in restricted area', 'unattended package', 'drone' as indicators of medium threat.
  If no such indicators are present, the threat is low.
  Return your assessment in the specified JSON format.
  `,
});

const assessThreatLevelFlow = ai.defineFlow(
  {
    name: 'assessThreatLevelFlow',
    inputSchema: ThreatAssessmentInputSchema,
    outputSchema: ThreatAssessmentOutputSchema,
  },
  async input => {
    // The LLM will now determine the threat level and description based on the scene description.
    const {output} = await prompt(input);
    if (!output) {
      // Fallback or error handling if the prompt fails to return structured output
      return {
        threatLevel: 'low',
        description: 'Could not assess threat due to an issue with the AI model.'
      };
    }
    return output;
  }
);
