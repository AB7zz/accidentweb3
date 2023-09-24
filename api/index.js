import express from 'express'
import bodyParser from 'body-parser'
import {ThirdwebSDK} from '@thirdweb-dev/sdk'
import dotenv from 'dotenv'
import { Configuration, OpenAIApi } from "openai";
import cors from 'cors'
import { createHelia } from 'helia'
import { json } from '@helia/json'

dotenv.config()

const app = express();
app.use(express.json({ limit: '10mb' }));
const port = process.env.PORT || 3000;
app.use(cors())

app.use(bodyParser.json());

const sdk = ThirdwebSDK.fromPrivateKey(`0x${process.env.PRIVATE_KEY}`, "mumbai", {
  secretKey: process.env.SECRET_KEY
});

const helia = await createHelia()
const j = json(helia)

app.get('/', async(req, res) => {
  res.send('Hello World')
})

app.post("/addAccident", async (req, res) => {
  try {
    const { _loc, _date, _time, snapShot, _plate } = req.body;
    let _snapShot = await j.add({ base64: snapShot })
    _snapShot = _snapShot.toString()
    
  const contract = await sdk.getContract(process.env.CONTRACT_ADDRESS);

  const result = await contract.call("addAccident", [_loc, _date, _time, _snapShot, _plate]);

  console.log(result)


    res.json({ success: true, result });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: error.message });
  }
});


app.post("/userAddsAccident", async (req, res) => {
  try {
    const { snapShot, _loc, _user } = req.body;
    const {latitude, longitude} = _loc
    let _snapShot = await j.add({ base64: snapShot })
    _snapShot = _snapShot.toString()

    // const sdk = ThirdwebSDK.fromWallet(_user, "mumbai", {
    //   secretKey: process.env.SECRET_KEY
    // });

    const contract = await sdk.getContract(process.env.CONTRACT_ADDRESS);

    const result = await contract.call("userAddsAccident", [_snapShot, latitude, longitude]);

    console.log(result)


    res.json({ success: true, result });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/getUserAccidents', async (req, res) => {
  try {
    const contract = await sdk.getContract(process.env.CONTRACT_ADDRESS);

    const result = await contract.call("getUserAccidents");

    console.log(result)


    res.json({ success: true, result });

  } catch (error) {
    console.log(error)
  }
})

app.get('/getAccidents', async (req, res) => {
    try {
        const contract = await sdk.getContract(process.env.CONTRACT_ADDRESS)
        const result = await contract.call("getAccidents")


        const arrayofMaps = result.map(async(innerArray) => {
          const map = {};
          const base64 = await j.get(innerArray[3])
          map['loc'] = innerArray[0]
          map['date'] = innerArray[1]
          map['time'] = innerArray[2]
          map['snapShot'] = base64.base64
          map['plate'] = innerArray[4]
          return map;
        });

        console.log(arrayofMaps)
        res.json({success: true, result: arrayofMaps})
    } catch (error) {
        console.log(error)
    }
})

app.get('/getAccident/:id', async (req, res) => {
    try {
        const contract = await sdk.getContract(process.env.CONTRACT_ADDRESS)
        const result = await contract.call("getAccident", [req.params.id])


          const arrayOfMap = {};
          const base64  = await j.get(result[3])
          arrayOfMap['loc'] = result[0]
          arrayOfMap['date'] = result[1]
          arrayOfMap['time'] = result[2]
          arrayOfMap['snapShot'] = base64.base64
          arrayOfMap['plate'] = result[4]

          console.log(arrayOfMap)

        res.json({success: true, result: arrayOfMap})
    } catch (error) {
        console.log(error)
    }
})


//--------------Dheeraj
app.post('/bot', async (req, res) => {
    try {
      const userMessage = req.body.message;
      console.log("userMessage",userMessage)
      const configuration = new Configuration({
        apiKey: process.env.CHAT_GPT_API,
      });
  
      const openai = new OpenAIApi(configuration);
      const prompt =`"Hello, I need legal assistance related to the Indian Penal Code (IPC) and other Indian laws. My problem is ${userMessage}. Can you provide me with guidance or information on how to approach this situation within the boundaries of Indian law?"`
      const completion = await openai.createCompletion({
        model: "text-davinci-003",
        prompt: userMessage, 
        temperature: 0,
        max_tokens: 2048,
      });
      const output = completion.data.choices[0].text.trim()
      console.log(output);
      
      // Send
      res.json({ output }); 
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Internal server error' }); 
    }
  });

app.listen(port, () => {
    console.log(`Server is running on port http://localhost:${port}`);
});