import React, {useEffect, useState} from "react";
import axios from "axios";
import {ImageData} from "../../Interfaces";

const FetchedImage: React.FC= () => {
    const EC2_BASE_URL = "http://34.231.244.123:5000"; // Ensure the protocol is included

    const [imageSrc, setImageSrc] = useState<string | null>(null);

    const displayImage = async (): Promise<void> => {
        const response = await fetch(`${EC2_BASE_URL}/api/images/image_batch0.png`);
        console.log(response)
        const imageBlob = await response.blob();
        const imageObjectURL = URL.createObjectURL(imageBlob);
        setImageSrc(imageObjectURL);
    };


    useEffect(() => {
        displayImage().then();
    }, []);

    return (
        <>
            {imageSrc && <img src={imageSrc} alt="Downloaded" />}
        </>
    );
}

export default FetchedImage;
