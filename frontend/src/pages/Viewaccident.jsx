import React,{useState,useEffect} from 'react';
import axios from 'axios'
function ViewAccident() {
  
  const [acc,setAcc] = useState([])
  
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiUrl = 'https://hackcdent-891d0b31e96e.herokuapp.com/getAccidents'
        const res = await axios.get(apiUrl)
        if(res){
          setAcc(res.data.result);
          console.log(res)
        }

      
      } catch (error) {
        console.error('Fetch Error:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="relative overflow-x-auto shadow-md sm:rounded-lg mt-[5rem] ml-[-6rem] mr-[4rem]">
      <table className="w-full text-sm text-left text-gray-500 dark:text-gray-400">
        <thead className="text-xl text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
          <tr>
            <th scope="col" className="px-6 py-3">
              Date
            </th>
            <th scope="col" className="px-6 py-3">
              Time
            </th>
            <th scope="col" className="px-6 py-3">
              Snapshot
            </th>
            <th scope="col" className="px-6 py-3">
              Location
            </th>
          </tr>
        </thead>
        <tbody>
          {acc && acc.map((product, index) => (
            <tr key={index} className={(index % 2 === 0) ? "bg-white border-b dark:bg-gray-900 dark:border-gray-700" : "border-b bg-gray-50 dark:bg-gray-800 dark:border-gray-700"}>
              <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
                {product.date}
              </th>
              <td className="px-6 py-4">
                {product.time}
              </td>
              <td className="px-6 py-4">
                {/* {product.snapShot} */
                console.log("product nte snapshot",product.snapShot)
                }
                <img src={product.snapShot} alt='random'/>
              </td>
              <td className="px-6 py-4">
                {product.loc}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ViewAccident;