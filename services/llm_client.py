import grpc
import llm_service_pb2
import llm_service_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        # Test prompt
        prompt = "Once upon a time"
        request = llm_service_pb2.InferenceRequest(prompt=prompt)
        
        # Send the request and get the response
        response = stub.Infer(request)
        print("LLM Response:", response.result)

if __name__ == '__main__':
    run()
