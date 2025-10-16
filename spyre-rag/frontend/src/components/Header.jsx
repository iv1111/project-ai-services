import React, { useState, useEffect } from "react";
// import { Link } from 'react-router-dom';

import {
  Header,
  HeaderName,
  HeaderGlobalBar,
  Select,
  SelectItem,
} from "@carbon/react";


const HeaderNav = () => {

  return (
    <Header aria-label="">
    <HeaderName to="/" prefix="">
        It Desk Support &#8482; Agent Dashboard
    </HeaderName>
    </Header>
  );
};

export default HeaderNav;
