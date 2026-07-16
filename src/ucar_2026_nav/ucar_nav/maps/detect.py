import cv2

def decode_qr_code(image_path):

    try:

        img = cv2.imread(image_path)
        if img is None:
            print(f"{image_path}")
            return []


        detector = cv2.QRCodeDetector()


        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)
        print(retval, decoded_info)



    except Exception as e:
        print(f"{e}")
        return []

if __name__ == "__main__":
    image_file = "IMG20250411164016.jpg" 
    decoded_data_list = decode_qr_code(image_file)

