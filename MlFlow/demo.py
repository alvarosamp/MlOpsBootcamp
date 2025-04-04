import os
import mlflow
import argparse
import time



def eval(p1,p2):
    output_metric = p1**2 + p2**2
    return output_metric


def main(inp1, inp2):
    with mlflow.start_run():
        mlflow.set_tag("version", "1.0.0")
        mlflow.log_param("input1", inp1)
        mlflow.log_param("input2", inp2)
        mlflow.log_metric("accuracy", 0.9)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", type=float, default=1.0, help="First input parameter")
    parser.add_argument("--input2", type=float, default=1.0, help="Second input parameter")
    args = parser.parse_args()  # Corrigir aqui para processar os argumentos

    metric = eval(args.input1, args.input2)  # Acessar os argumentos corretamente
    mlflow.log_metric("metric", metric)
    os.makedirs("dummy", exist_ok=True)
    with open("dummy/metric.txt", "w") as f:
        f.write(f'Metric: {metric}')
    mlflow.log_artifact("dummy/metric.txt")


    #Parsed_args.param1