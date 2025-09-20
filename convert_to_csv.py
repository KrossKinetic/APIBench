import json
import csv
import argparse

'''
Converts APIBench-Q json at "./Python_Queries/OriginalPythonQueries.json" 
to "./Python_Queries/Python_Queries_Formatted.csv"
by default, can be changed to anything, including java. A preformated copy of both are available in their 
respective directories.

Parameters = input_file, output_file
'''

def run_main(path_in:str,path_out:str):
    data_to_write = [["Query","APIs","APIClasses"]]
    with open(path_in,"r") as f:
        json_file = json.load(f)
        for i in json_file.values():
            
            if "APIs" not in i or "OriginalQuery" not in i or "APIClasses" not in i:
                continue
            
            all_apis = ""
            for j in i["APIs"]:
                all_apis+=j+"; "

            all_classes = ""
            for k in i["APIClasses"]:
                all_classes+=k+"; "

            data_to_write.append([i["OriginalQuery"],all_apis,all_classes])


    with open(path_out,"w") as f:
        writer_obj = csv.writer(f)
        writer_obj.writerows(data_to_write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two command-line arguments.")
    
    parser.add_argument("input_file", nargs="?", default="./Python_Queries/OriginalPythonQueries.json", help="Path to the input file")
    parser.add_argument("output_file", nargs="?", default="./Python_Queries/Python_Queries_Formatted.csv", help="Path to the output file")
    
    args = parser.parse_args()

    run_main(args.input_file,args.output_file)