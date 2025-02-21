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
        # loads .env variables
        load_dotenv()
        global bucket_name, access_key, access_secret, region

        bucket_name = os.environ['BUCKET_NAME']
        access_key = os.environ['ACCESS_KEY']
        access_secret = os.environ['SECRET_ACCESS_KEY']
        region = os.environ['BUCKET_REGION']
        s3_link = 'http://{}.s3.amazonaws.com/'.format(bucket_name)


        s3_client = boto3.client(
            "s3", 
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=access_secret
        )

        try:    
            s3_client.upload_fileobj(
                buffer, 
                bucket_name, 
                img_name, 
                ExtraArgs={
                    "ContentType": img_type,
                }
            )

        except Exception as err:

            return err

        
    

