exports.handler = async (event) => {
  const code = event.queryStringParameters?.code;
  
  if (!code) {
    const params = new URLSearchParams({
      client_id: process.env.DISCORD_CLIENT_ID,
      redirect_uri: 'https://splashplaypodcast.com/.netlify/functions/discord-auth',
      response_type: 'code',
      scope: 'identify guilds.members.read'
    });
    return {
      statusCode: 302,
      headers: { Location: `https://discord.com/api/oauth2/authorize?${params}` }
    };
  }
  
  try {
    const tokenRes = await fetch('https://discord.com/api/oauth2/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: process.env.DISCORD_CLIENT_ID,
        client_secret: process.env.DISCORD_CLIENT_SECRET,
        grant_type: 'authorization_code',
        code,
        redirect_uri: 'https://splashplaypodcast.com/.netlify/functions/discord-auth'
      })
    });
    const tokens = await tokenRes.json();
    if (!tokens.access_token) throw new Error('No access token');
    
    const userRes = await fetch('https://discord.com/api/users/@me', {
      headers: { Authorization: `Bearer ${tokens.access_token}` }
    });
    const user = await userRes.json();
    
    const memberRes = await fetch(
      `https://discord.com/api/users/@me/guilds/${process.env.GUILD_ID}/member`,
      { headers: { Authorization: `Bearer ${tokens.access_token}` } }
    );
    const member = await memberRes.json();
    const roles = member.roles || [];
    
    let tier = 'free';
    const adminRole = process.env.ADMIN_ROLE_ID;
    const vipRoles = [process.env.VIP_ROLE_ID_1, process.env.VIP_ROLE_ID_2];
    const squirtRoles = [process.env.SQUIRT_ROLE_ID_1, process.env.SQUIRT_ROLE_ID_2];
    
    if (roles.includes(adminRole)) tier = 'admin';
    else if (vipRoles.some(r => roles.includes(r))) tier = 'vip';
    else if (squirtRoles.some(r => roles.includes(r))) tier = 'squirt';
    
    const cookieValue = Buffer.from(JSON.stringify({ 
      name: user.username, 
      role: tier,
      id: user.id 
    })).toString('base64');
    
    return {
      statusCode: 302,
      headers: {
        Location: '/nba',
        'Set-Cookie': `sp_discord=${cookieValue}; Path=/; Secure; SameSite=Lax; Max-Age=604800`
      }
    };
  } catch (err) {
    console.error(err);
    return {
      statusCode: 302,
      headers: { Location: '/nba?auth_error=1' }
    };
  }
};
