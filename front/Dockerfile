FROM nginx:latest
# delete the nginx configuration file 
RUN rm /etc/nginx/conf.d/default.conf
# copy our own configuration file
COPY nginx.conf /etc/nginx/conf.d/

COPY static/css /usr/share/nginx/html/static/css
COPY templates/ /usr/share/nginx/html/templates/

EXPOSE 80
CMD ["nginx","-g", "daemon off;"]