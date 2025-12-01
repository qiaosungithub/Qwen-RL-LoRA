import os
import json
from typing import Dict, List

class VerifierClient:
    """
    Verifier client for CAPO credit assignment.
    Analyzes reasoning steps to identify errors.
    """
    def __init__(self, config):
        self.api_type = config.capo.verifier_api_type
        self.model_name = config.capo.verifier_model
        self.max_steps = config.capo.max_steps
        
        # Metrics tracking
        self.total_calls = 0
        self.failed_calls = 0
        
        if self.api_type == 'openai':
            import openai
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.api_type == 'qwen':
            # Placeholder for Qwen API
            pass
        elif self.api_type == 'dummy':
            # No initialization needed for dummy mode
            pass
        else:
            raise ValueError(f"Unknown verifier API type: {self.api_type}")
    
    def verify_reasoning(self, question: str, response: str, ground_truth: int) -> Dict:
        """
        Verify reasoning steps and identify errors.
        
        Args:
            question: The problem statement
            response: Model's chain-of-thought response
            ground_truth: Correct answer
            
        Returns:
            {
                'wrong_steps': [int],  # 0-indexed list of wrong step indices
                'total_steps': int,     # Total number of steps
                'explanation': str,     # Debug info
                'success': bool         # Whether verification succeeded
            }
        """
        self.total_calls += 1
        
        if self.api_type == 'dummy':
            return self._dummy_verify(response, ground_truth)
        elif self.api_type == 'openai':
            return self._openai_verify(question, response, ground_truth)
        elif self.api_type == 'qwen':
            return self._qwen_verify(question, response, ground_truth)
        else:
            return self._fallback_result()
    
    def _dummy_verify(self, response: str, ground_truth: int) -> Dict:
        """Dummy verifier for testing without API costs."""
        # Split by newlines to count steps
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        total_steps = len(lines)
        
        # Parse answer to determine if response is correct
        from utils.reward_util import parse_answer
        predicted = parse_answer(response)
        
        if predicted == ground_truth:
            # Correct answer - no wrong steps
            wrong_steps = []
        else:
            # Wrong answer - mark last 2 steps as wrong (arbitrary for testing)
            wrong_steps = list(range(max(0, total_steps - 2), total_steps))
        
        return {
            'wrong_steps': wrong_steps,
            'total_steps': total_steps,
            'explanation': f'Dummy verification: {len(wrong_steps)} wrong steps',
            'success': True
        }
    
    def _openai_verify(self, question: str, response: str, ground_truth: int) -> Dict:
        """Use OpenAI API to verify reasoning steps."""
        try:
            prompt = f"""Analyze the following mathematical reasoning and identify which steps contain errors.

Question: {question}
Correct Answer: {ground_truth}

Student's Reasoning:
{response}

Please:
1. Split the reasoning into individual steps (separated by newlines)
2. Identify which steps contain mathematical errors or logical mistakes
3. Return a JSON object with:
   - "wrong_steps": list of 0-indexed step numbers that are wrong
   - "total_steps": total number of steps
   - "explanation": brief explanation of errors

Output only valid JSON, no other text."""

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            
            result_text = completion.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            return {
                'wrong_steps': result.get('wrong_steps', []),
                'total_steps': result.get('total_steps', 0),
                'explanation': result.get('explanation', ''),
                'success': True
            }
            
        except Exception as e:
            print(f"[Verifier] OpenAI API error: {e}")
            self.failed_calls += 1
            return self._fallback_result()
    
    def _qwen_verify(self, question: str, response: str, ground_truth: int) -> Dict:
        """Placeholder for Qwen API verification."""
        # TODO: Implement Qwen API verification
        print("[Verifier] Qwen API not implemented yet, using fallback")
        self.failed_calls += 1
        return self._fallback_result()
    
    def _fallback_result(self) -> Dict:
        """Fallback when verification fails."""
        return {
            'wrong_steps': [],
            'total_steps': 0,
            'explanation': 'Verification failed, using fallback',
            'success': False
        }
    
    @property
    def metrics(self) -> Dict:
        """Return verification metrics."""
        return {
            'total_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'fallback_rate': self.failed_calls / max(1, self.total_calls)
        }
