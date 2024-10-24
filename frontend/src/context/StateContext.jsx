import React, {useContext, createContext} from 'react'
import {useAddress, useMetamask} from '@thirdweb-dev/react'

const StateContext = createContext()

export const StateContextProvider = ({children}) => {
    const address = useAddress()
    const connect = useMetamask()
    const apiUrl = 'https://hackcdent-891d0b31e96e.herokuapp.com'

    return(
        <StateContext.Provider value={{
            connect,
            address,
            apiUrl
        }}>
            {children}
        </StateContext.Provider>
    )
}

export const useStateContext = () => useContext(StateContext)