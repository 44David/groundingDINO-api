from dotenv import load_dotenv
import os
import boto3, botocore
import io
from PIL import Image
from botocore.exceptions import ClientError


class Upload():

    def create_presigned_url(self, obj_name):

        s3_client = boto3.client(
            "s3", 
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=access_secret
        )
        expiration = 60
        
        try:
            response = s3_client.generate_presigned_url('get_object', Params={ 'Bucket': bucket_name, 'Key': obj_name }, ExpiresIn=expiration)
            
        except ClientError as e:

            return e

        return response
     
    def s3_upload(self, buffer, img_name, img_type):
        global bucket_name, access_key, access_secret, region


        client = boto3.client('ssm', region_name='us-east-1')


        bucket_name = client.get_parameter(Name='bucket_name', WithDecryption=True)
        access_key = client.get_parameter(Name='access_key', WithDecryption=True)
        access_secret = client.get_parameter(Name='access_secret', WithDecryption=True)
        region = client.get_parameter(Name='bucket_region', WithDecryption=True)

        s3_link = 'http://{}.s3.amazonaws.com/'.format(bucket_name)


        s3_client = boto3.client(
            "s3", 
            region_name=region['Parameter']['Value'],
            aws_access_key_id=access_key['Parameter']['Value'],
            aws_secret_access_key=access_secret['Parameter']['Value']
        )

        try:    
            s3_client.upload_fileobj(
                buffer, 
                bucket_name['Parameter']['Value'], 
                img_name, 
                ExtraArgs={
                    "ContentType": img_type,
                }
            )

        except Exception as err:

            return err

        
    

