import grpc
from concurrent import futures
from transformers import pipeline
from llm_service_pb2 import InferenceResponse
import llm_service_pb2_grpc

class LLMServiceServicer(llm_service_pb2_grpc.LLMServiceServicer):
    def __init__(self):
        # Load the GPT-2 model
        self.generator = pipeline('text-generation', model='gpt2')

    def Infer(self, request, context):
        # Generate text with controlled parameters
        result = self.generator(
            request.prompt,
            max_length=500,  # Limit response length
            temperature=0.7,  # Control randomness
            top_k=500,  # Limit to top 50 words
            top_p=0.9,  # Use nucleus sampling
            num_return_sequences=1  # Generate only one response
        )[0]['generated_text']
        return InferenceResponse(result=result)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    llm_service_pb2_grpc.add_LLMServiceServicer_to_server(LLMServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("🚀 LLM Server is running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
